from tqdm import tqdm
from PIL import Image
import geopandas as gp
import numpy as np
from src.tiles import tiles_from_slippy_map
from src.features.building import Roof_features

def mask_to_feature(input_mask, output_pv_feature):
    handler = Roof_features()
    reults = list(tiles_from_slippy_map(input_mask))

    for tile, path in tqdm(reults, ascii=True, unit="mask"):
        image = np.array(Image.open(path).convert("P"), dtype=np.uint8)
        mask = (image == 1).astype(np.uint8)
        handler.apply(tile, mask)

    feature = handler.jsonify()
    feature = gp.GeoDataFrame.from_features(feature, crs=4326)
    feature.to_file(output_pv_feature, driver='GeoJSON')
    print('Finish resluts_to_feature')
    return feature

def intersect(input_mask, output_pv_feature, Bld_footprints, Bld_with_pv):

    pv_polygons = mask_to_feature(input_mask, output_pv_feature)
    Bld_polygons = gp.GeoDataFrame.from_file(Bld_footprints)[['geometry']]
    Bld_polygons['area'] = Bld_polygons['geometry'].to_crs({'init': 'epsg:4326'}).map(lambda p: p.area)

    intersections = gp.sjoin(Bld_polygons, pv_polygons, how="inner", op='intersects')
    intersections = intersections.drop_duplicates(subset=['geometry'])
    intersections.to_file(Bld_with_pv, driver='GeoJSON')


if __name__ == '__main__':
    input_mask = './prediction_demo/output'   # INPUT: model prediction results (raster masks)
    output_pv_feature = './prediction_demo/mask_to_features' + ".geojson"    # OUTPUT: pv polygons (vector features)
    Bld_footprints = 'Bld_footprints.geojson'    # INPUT: building footprints (vector features)
    Bld_with_pv = 'Bld_with_pv.geojson'    # OUTPUT: building footprints with pv panels (vector features)

    intersect(input_mask, output_pv_feature, Bld_footprints, Bld_with_pv)