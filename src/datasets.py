"""PyTorch-compatible datasets.

Guaranteed to implement `__len__`, and `__getitem__`.

See: http://pytorch.org/docs/0.3.1/data.html
"""

import torch
import numpy
from PIL import Image
import torch.utils.data
import torch.utils.data as data
import os
from src.transforms import ConvertImageMode, ImageToTensor

from src.tiles import tiles_from_slippy_map, buffer_tile_image


# Single Slippy Map directory structure
class SlippyMapTiles(torch.utils.data.Dataset):
    """Dataset for images stored in slippy map format.
    """

    def __init__(self, root, transform=None):
        super(SlippyMapTiles,).__init__()

        self.tiles = []
        self.transform = transform

        self.tiles = [(tile) for tile in tiles_from_slippy_map(root)]
        self.tiles.sort(key=lambda tile: tile[0])

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        tile = self.tiles[i]
        path = os.path.join(root, tile)
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, tile

def load_img(path):
    image = Image.open(path)
    return image

# Multiple Slippy Map directories.
# Think: one with images, one with masks, one with rasterized traces.
class SlippyMapTilesConcatenation(torch.utils.data.Dataset):
    """Dataset to concate multiple input images stored in slippy map format.
    """

    def __init__(self, inputs, target, joint_transform=None):
        super(SlippyMapTilesConcatenation, self).__init__()

        # No transformations in the `SlippyMapTiles` instead joint transformations in getitem
        self.joint_transform = joint_transform

        # self.inputs = [SlippyMapTiles(inp) for inp in inputs]
        # self.target = SlippyMapTiles(target)
        self.inputs = [os.path.join(inputs, x) for x in os.listdir(inputs)]
        self.target = [os.path.join(target, x) for x in os.listdir(target)]
        
        # assert len(set([len(dataset) for dataset in self.inputs])) == 1, "same number of tiles in all images"
        assert len(self.target) == len(self.inputs), "same number of tiles in images and label"

    def __getitem__(self, i):
        # at this point all transformations are applied and we expect to work with raw tensors
        # inputs = [dataset[i] for dataset in self.inputs]
        
        # images = [image for image, _ in inputs]
        images = [load_img(self.inputs[i])]
        mask = load_img(self.target[i])
        
        tiles = [tile for tile in self.inputs]
        
        # mask, mask_tile = self.target[i]

        # assert len(set(tiles)) == 1, "all images are for the same tile"
        # assert tiles[0] == mask_tile, "image tile is the same as label tile"

        if self.joint_transform is not None:
            images, mask = self.joint_transform(images, mask)
        mask[mask!=0] = 1

        return torch.cat(images, dim=0), mask, tiles

    def __len__(self):
        return len(self.target)


# Todo: once we have the SlippyMapDataset this dataset should wrap
# it adding buffer and unbuffer glue on top of the raw tile dataset.
class BufferedSlippyMapDirectory(torch.utils.data.Dataset):
    """Dataset for buffered slippy map tiles with overlap.
    """

    def __init__(self, root, mask_root, transform=None, size=512, overlap=32):
        """
        Args:
          root: the slippy map directory root with a `z/x/y.png` sub-structure.
          transform: the transformation to run on the buffered tile.
          size: the Slippy Map tile size in pixels
          overlap: the tile border to add on every side; in pixel.

        Note:
          The overlap must not span multiple tiles.

          Use `unbuffer` to get back the original tile.
        """

        super().__init__()

        assert overlap >= 0
        assert size >= 256

        self.transform = transform
        self.size = size
        self.root = root
        self.mask_root = mask_root
        self.overlap = overlap
        self.tiles = [tile for tile in os.listdir(root)]
        self.tiles.sort(key=lambda tile: tile[0])

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        # tile, path = self.tiles[i]
        tile = self.tiles[i]
        path = os.path.join(self.root, tile)
        # image = buffer_tile_image(tile, self.tiles, overlap=self.overlap, tile_size=self.size)
        image = Image.open(path)
        target = Image.open(os.path.join(self.mask_root, tile)).convert('P')
        if self.transform is not None:
            image = self.transform(image)
            target = torch.tensor(numpy.array(target))
        return image, target, tile#, torch.IntTensor([tile.x, tile.y, tile.z])

    def unbuffer(self, probs):
        """Removes borders from segmentation probabilities added to the original tile image.

        Args:
          probs: the segmentation probability mask to remove buffered borders.

        Returns:
          The probability mask with the original tile's dimensions without added overlap borders.
        """

        o = self.overlap
        _, x, y = probs.shape

        return probs[:, o : x - o, o : y - o]
