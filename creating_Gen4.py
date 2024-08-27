import json
import shutil
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import xml.etree.ElementTree as ET
import uuid
import datetime
from shapely.geometry import LineString, Point
import unicodedata
import zipfile
from shapely.ops import transform
import pyproj
import logging
import sys
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore")


def now_time(john_deere=True):
    """At this time in string"""

    if john_deere:
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        now = datetime.datetime.now()
        date_time = now.strftime("%H:%M:%S")

    return date_time


def my_print(str_my_print):
    print(str_my_print)
    return str_my_print


def unique_id(name):
    """Universally Unique Identifier (UUID), also known as a Globally Unique Identifier (GUID).
    UUIDs/GUIDs are 128-bit numbers used to uniquely identify information in computer systems."""

    # Namespace for DNS (you can also use other namespaces like uuid.NAMESPACE_URL, uuid.NAMESPACE_OID, etc.)
    namespace = uuid.NAMESPACE_DNS

    # Generate a UUID based on the SHA-1 hash of a namespace identifier and a name.
    result_uuid = str(uuid.uuid5(namespace, name))

    return result_uuid


def diacritic(my_chain):
    """Removed diacritics and replaced spaces with underscores"""
    bad_chars = [r' ', r',', r'-', r'&', r'[', r']', r'(', r')', r'__', r'/', r'.', r'+', r' - ']

    for bad_char in bad_chars:
        if bad_char in my_chain:
            my_chain = my_chain.replace(bad_char, '_')

    my_chain = unicodedata.normalize('NFKD', my_chain)

    output = ""
    for letter in my_chain:
        if not unicodedata.combining(letter):
            output += letter

    return output


def save_element_xml(element, path):
    """Save elements to file .xml"""

    el = ET.ElementTree(element)
    el.write(path)


def element_parse(path_xml):
    xml_tree = ET.parse(path_xml)
    root = xml_tree.getroot()

    return root


def list_element_parse(path_xml, name_element=''):
    xml_tree = ET.parse(path_xml)
    root = xml_tree.getroot()

    element_list = [x for x in root.findall(name_element)][0]

    return element_list


def zip_folder(folder_path, zip_name):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, os.path.dirname(folder_path)))


def convert_coors(geometry, from_wgs84_to_utm33n=False, from_utm33n_to_wgs84=False):
    """Convert coordinate systek"""

    # Conver WGS84 to UTM33N and back
    wgs84 = pyproj.CRS('EPSG:4326')
    utm = pyproj.CRS('EPSG:32633')
    project_UTM33 = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    project_WGS84 = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform

    if from_wgs84_to_utm33n:
        return transform(project_UTM33, geometry)
    elif from_utm33n_to_wgs84:
        return transform(project_WGS84, geometry)
    else:
        geometry


def split_line_distance(line, distance):
    """Split line base your distance"""

    # Convert to UTM33N
    line_utm33 = convert_coors(line, from_wgs84_to_utm33n=True)

    # Set lenght and
    lenght = line_utm33.length
    list_line_utm33 = list(line_utm33.coords)

    # Split by distance
    if len(list_line_utm33) > 2 and distance < lenght:

        # Sum lenght by distance
        i = lenght // distance
        i_array = np.arange(1, i + 1, 1)
        i_array = i_array * distance

        # Interpolate distance
        list_point_distance = [Point(list_line_utm33[0])]
        for x_dist in i_array:
            x_point = line_utm33.interpolate(x_dist)
            list_point_distance.append(x_point)

        list_point_distance.append(Point(list_line_utm33[-1]))
        line_new_utm33n = LineString(list_point_distance)
    else:
        line_new_utm33n = LineString(list_line_utm33)

    # Convert to wgs84
    line_new_wgs84 = convert_coors(line_new_utm33n, from_utm33n_to_wgs84=True)

    # Load to DataFrame
    line_new_wgs84_coords = list(line_new_wgs84.coords)
    line_new_wgs84_coords = [[x] for x in line_new_wgs84_coords]
    df = gpd.GeoDataFrame(columns=["VERTEX"], data=line_new_wgs84_coords)

    if df.shape[0] >= 3:
        # Point array shift 1
        df["SHIFT_1"] = df["VERTEX"].shift(periods=1, fill_value=0)
        df["SHIFT_1"].loc[0] = df["VERTEX"].loc[0]

        # Point array shift 2
        df["SHIFT_2"] = df["VERTEX"].shift(periods=2, fill_value=0)
        df["SHIFT_2"].loc[0], df["SHIFT_2"].loc[1] = df["VERTEX"].loc[0], df["VERTEX"].loc[0]

        # Calculate angle and distance
        df["ANGLE"] = df.apply(lambda x: abs(get_angle(x["VERTEX"], x["SHIFT_1"], x["SHIFT_2"])), axis=1)
        df["ANGLE"] = df["ANGLE"].shift(periods=-1, fill_value=0)
        df["ANGLE"].loc[0] = 0
        df["ANGLE"].loc[df.shape[0] - 1] = 0

        # If angle in 180 thaan verxtex remove
        df = df.loc[df["ANGLE"] < 179.999]

        # Convert to line
        line_new_wgs84 = LineString(df["VERTEX"].to_list())

    return line_new_wgs84


def split_lines_angle(line, low_angle_limit):
    """Create line by angle"""

    vertex = list(line.coords)
    vertex = [[x] for x in vertex]
    df = gpd.GeoDataFrame(data=vertex, columns=["VERTEX"])
    df = df.reset_index()

    # Point array shift 1
    df["SHIFT_1"] = df["VERTEX"].shift(periods=1, fill_value=0)
    df["SHIFT_1"].loc[0] = df["VERTEX"].loc[0]

    # Point array shift 2
    df["SHIFT_2"] = df["VERTEX"].shift(periods=2, fill_value=0)
    df["SHIFT_2"].loc[0], df["SHIFT_2"].loc[1] = df["VERTEX"].loc[0], df["VERTEX"].loc[0]

    # Calculate angle
    df["ANGLE"] = df.apply(lambda x: abs(get_angle(x["VERTEX"], x["SHIFT_1"], x["SHIFT_2"])), axis=1)
    df["ANGLE"] = df["ANGLE"].shift(periods=-1, fill_value=0)  # Puvodne 180, ale je spravne 0
    df["ANGLE_IF"] = df["ANGLE"].apply(lambda x: 1 if x > low_angle_limit else 0)
    df["ANGLE_IF"].loc[-1] = 1

    # Select base on angle limit
    df_inside = df.iloc[1:-1]
    df_inside_false = df_inside.loc[df_inside["ANGLE_IF"] == 0]

    # If there is no angle limit
    dict_line = {}
    if not df_inside_false.shape[0]:
        dict_line[0] = LineString(df["VERTEX"].to_list())
        return dict_line

    first_index = df_inside_false.loc[df_inside_false.index[0], 'index']
    df_final = df_inside.loc[df_inside["index"] >= first_index]

    df_final = df_final.append(df.iloc[-1])
    df_final = df_final.append(df_inside.loc[df_inside["index"] <= first_index])
    df_final["index_new"] = np.arange(0, df_final.shape[0])

    df_final["geometry"] = df_final.apply(lambda x: Point(x["VERTEX"]), axis=1)
    df_final = gpd.GeoDataFrame(df_final)

    # Select point lass than angle limit
    df_false = df_final[df_final["ANGLE_IF"] == 0]
    last_id = df_false["index_new"].to_list()[-1]
    df_false["FID_SHIFT"] = df_false["index_new"].shift(periods=-1, fill_value=last_id)

    # Calculate distance and select distance more than my limit distance
    df_final["DISTANCE"] = df.apply(lambda x: my_distance(x["SHIFT_1"], x["VERTEX"]), axis=1)
    for idn, row in df_false.iterrows():

        df_split = df_final[(df_final["index_new"] >= row["index_new"]) &
                            (df_final["index_new"] <= row["FID_SHIFT"])]

        try:
            df_split.loc[df_split.index[0], "DISTANCE"] = 0
        except IndexError:
            continue

        if df_split.shape[0] > 1:
            dict_line[idn] = LineString(df_split["VERTEX"].to_list())

    return dict_line


def my_distance(point1, point2):
    """Calculate disntace base on from WGs84 to UTM33N"""

    point1 = Point(point1)
    point2 = Point(point2)

    # Project
    point1_utm33 = convert_coors(point1, from_wgs84_to_utm33n=True)
    point2_utm33 = convert_coors(point2, from_wgs84_to_utm33n=True)

    # Calculate
    distance_meters = point1_utm33.distance(point2_utm33)

    return distance_meters


def get_angle(p0, p1=np.array([0, 0]), p2=None):
    ''' compute angle (in degrees) for p0p1p2 corner
    Inputs:
    p0,p1,p2 - points in the form of [x,y]
    '''

    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return np.degrees(angle)


def check_column(colm_name, df):
    """Check if column is not exist"""

    if colm_name in df.columns.to_list():
        return df
    else:
        df[colm_name] = colm_name
        return df


def gdf_multi_single_line(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert MultiLineString to LineString in GeoDataFrame and calculate count of points and length of line"""

    if df.shape[0] == 0:
        return df

    list_geo = []
    for x in range(0, df.shape[0]):
        row = df.iloc[x]
        if row.geometry.geom_type == 'MultiLineString':
            for x_geom in row.geometry.geoms:
                list_geo.append(list(row)[:-1] + [x_geom])
        elif row.geometry.geom_type == 'LineString':
            list_geo.append(list(row))
        elif row.geometry.geom_type == 'GeometryCollection':
            line_row = [x for x in list(row.geometry.geoms) if x.geom_type == 'LineString']
            for x_geom in line_row:
                list_geo.append(list(row)[:-1] + [x_geom])

    df_new = gpd.GeoDataFrame(columns=df.columns, data=list_geo, crs=4326)
    df_new['geometry'] = df_new['geometry'].to_crs(32633)
    df_new[['count_point', 'length']] = df_new['geometry'].apply(lambda x: pd.Series([len(x.coords), x.length]))
    df_new['geometry'] = df_new['geometry'].to_crs(4326)

    return df_new


def dict_line_types(type_number):
    """Check if types of line/curve:
     1 -> AB Line
     2 -> AB Curve
     3 -> Adaptive Line"""

    if 3 >= type_int >= 1:
        return type_int
    else:
        return 3


class Gen4FromShp:

    def __init__(self, json_input, output_log=[], processBar=None):

        self.output_log = output_log
        self.json_input = json_input
        self.base_fold = os.path.dirname(json_input['shpFields'])
        self.fold_jd_data = self.base_fold + '/JD data'
        if not os.path.exists(self.fold_jd_data):
            os.mkdir(self.fold_jd_data)

        # Creating folders
        self.fold_gen4 = os.path.join(self.fold_jd_data, "Gen4")
        if not os.path.exists(self.fold_gen4):
            os.mkdir(self.fold_gen4)

        self.fold_spatial = os.path.join(self.fold_gen4, "SpatialFiles")
        if not os.path.exists(self.fold_spatial):
            os.mkdir(self.fold_spatial)

        self.gdf_fields = gpd.read_file(json_input['shpFields'])
        if os.path.exists(json_input['shpGuide']):
            self.gdf_lines = gpd.read_file(json_input['shpGuide'])
        else:
            self.gdf_lines = gpd.GeoDataFrame(columns=["ID", "geometry"])

        self.client_column = json_input['clientName']
        self.farm_column = json_input['farmColumn']
        self.field_column = json_input['fieldColumn']
        self.crop_column = json_input['cropColumn']
        self.line_column = json_input['lineColumn']
        self.line_type_column = json_input['lineColumnType']
        if not len(self.crop_column):
            self.crop_column = 'Crop'

        self.master_data_path = os.path.join(self.fold_gen4, 'MasterData.xml')

        # Load json metadata
        with open(json_input['json_metadat'], 'r') as j:
            metadata = dict(json.load(j))

        self.metadata_element = metadata['element']
        self.bool_Guidance = True
        self.processBar = processBar
        self.id_process = now_time()
        self.dict_line_types = {
            "AB Line": 1,
            "AB Curve": 2,
            "Adaptive Curve": 3,
        }

    def prepare_gdf(self):

        """Prepare shapefile data"""

        # Control geometry of polygons and line
        try:
            self.gdf_fields.geometry = self.gdf_fields.geometry.set_crs(4326)
        except ValueError:
            self.gdf_fields.geometry = self.gdf_fields.geometry.to_crs(4326)
        try:
            self.gdf_lines.geometry = self.gdf_lines.geometry.set_crs(4326)
        except ValueError:
            self.gdf_lines.geometry = self.gdf_lines.geometry.to_crs(4326)

        self.gdf_lines = gpd.overlay(self.gdf_lines, self.gdf_fields, keep_geom_type=False)
        self.gdf_lines = gdf_multi_single_line(df=self.gdf_lines)
        self.gdf_lines = self.gdf_lines.reset_index()
        if self.line_column in self.gdf_lines.columns.to_list():
            self.gdf_lines[self.line_column] = self.gdf_lines.apply(
                lambda x: "Line" if isinstance(x[self.line_column], type(None)) else x[self.line_column], axis=1)
            self.gdf_lines["ID_line"] = self.gdf_lines.apply(
                lambda x: f"{x[self.line_column]}_{x['index'] + 1}", axis=1)
        else:
            self.gdf_lines["ID_line"] = self.gdf_lines.apply(lambda x: "Line" + str(x['index'] + 1), axis=1)

        ###############
        print("Unique: ", len(self.gdf_lines["ID_line"].unique()), "All: ", self.gdf_lines.shape)
        ###############

        self.gdf_fields['Client'] = self.client_column

        # Check if column Client, Farm, Crop is not exist
        self.gdf_fields = check_column(self.client_column, self.gdf_fields)
        self.gdf_fields = check_column(self.farm_column, self.gdf_fields)
        self.gdf_fields = check_column(self.crop_column, self.gdf_fields)

        # Check if column Line exist
        if not len(self.line_type_column):
            self.gdf_lines = self.gdf_lines.rename(columns={self.line_type_column: 'Type_line'})
            self.line_type_column = 'Type_line'
            self.gdf_lines[self.line_type_column] = self.dict_line_types['Adaptive Curve']

        self.gdf_lines[self.line_type_column] = self.gdf_lines[self.line_type_column].fillna(0)
        self.gdf_lines[self.line_type_column] = self.gdf_lines[self.line_type_column].astype(int)
        # If type line is not from 1 to 3 then set default 3 as Adaptive Curve
        self.gdf_lines.loc[(self.gdf_lines[self.line_type_column] <
                            self.dict_line_types['AB Line']), self.line_type_column] = self.dict_line_types[
            'Adaptive Curve']
        self.gdf_lines.loc[(self.gdf_lines[self.line_type_column] >
                            self.dict_line_types['Adaptive Curve']), self.line_type_column] = self.dict_line_types[
            'Adaptive Curve']

        if self.processBar:
            self.processBar.setValue(20)

    def creat_setup_xml(self):
        """Creat MasterData.xml with element Client, Farm, Field"""

        # Main root SetupFile
        root = ET.Element('SetupFile', {'xmlns:xsi': "http://www.w3.org/2001/XMLSchema-instance",
                                        'xmlns:xsd': "http://www.w3.org/2001/XMLSchema",
                                        'xmlns': "urn:schemas-johndeere-com:Setup"})

        # Element SourceApp
        element_list = element_parse(self.metadata_element['SourceApp'])
        root.append(element_list)

        # Element FileSchemaVersion
        element = element_parse(self.metadata_element['FileSchemaVersion'])
        root.append(element)

        # Element Setup
        elm_setup = ET.SubElement(root, 'Setup')
        columns_select = [self.client_column, self.farm_column, self.field_column, self.crop_column, 'geometry']

        # Client elements
        gdf_first_select = self.gdf_fields[columns_select].groupby(self.client_column, as_index=False).first()
        for i, x_row in gdf_first_select.iterrows():
            create_element = CreateElement(self.json_input,
                                           self.metadata_element,
                                           client=x_row[self.client_column],
                                           id_process=self.id_process)
            elm_setup = create_element.create_client(element=elm_setup)

        # Farm elements
        gdf_first_select = self.gdf_fields[columns_select].groupby(self.farm_column, as_index=False).first()
        for i, x_row in gdf_first_select.iterrows():
            create_element = CreateElement(self.json_input,
                                           self.metadata_element,
                                           client=x_row[self.client_column],
                                           farm=x_row[self.farm_column],
                                           id_process=self.id_process)
            elm_setup = create_element.create_farm(element=elm_setup)

        gdf_first_select = self.gdf_fields[columns_select].groupby(self.field_column, as_index=False).first()
        gdf_first_select = gdf_first_select.reset_index()

        i_max = gdf_first_select.shape[0]
        for i, x_row in gdf_first_select.iterrows():

            self.output_log.append(
                my_print(f'{now_time(john_deere=False)} - Field {i + 1}: {diacritic(x_row[self.field_column])}'))
            create_element = CreateElement(self.json_input,
                                           self.metadata_element,
                                           field=x_row[self.field_column],
                                           farm=x_row[self.farm_column],
                                           crop=x_row[self.crop_column],
                                           id_process=self.id_process)
            # Field element
            elm_setup = create_element.create_field(element=elm_setup)

            # Crop
            elm_setup = create_element.create_crop(element=elm_setup)

            # Boundary element
            elm_setup, path_boundary = create_element.create_boundary(element=elm_setup, idn=i)
            self.create_geojson_boundary(path_boundary, x_row['geometry'])

            gdf_first_select.geometry = gdf_first_select.geometry.set_crs(4326)
            x_gdf_first_select = gdf_first_select.loc[gdf_first_select['index'] == x_row['index']]
            x_gdf_line = gpd.overlay(df1=self.gdf_lines, df2=x_gdf_first_select, keep_geom_type=False)

            x_gdf_line.geometry = x_gdf_line.geometry.to_crs(32633)
            x_gdf_line["length"] = x_gdf_line.geometry.length
            x_gdf_line = x_gdf_line[x_gdf_line["length"] >= 1]
            x_gdf_line.geometry = x_gdf_line.geometry.to_crs(4326)

            # Guidance element
            if not x_gdf_line.empty:

                # Find first geometry of LineString, no MultiLineString
                try:
                    first_line = [x for x in x_gdf_line.geometry.to_list() if 'LineString' == x.geom_type][0]
                except IndexError:
                    list_multi = x_gdf_line.geometry.to_list()[0]
                    first_line = [x for x in list_multi.geoms][0]

                self.output_log.append(my_print(f'{now_time(john_deere=False)} - Create guidance'))
                lat, long = [list(x) for x in list(zip(first_line.xy[1], first_line.xy[0]))][0]
                elm_setup, self.bool_Guidance = create_element.create_guidance(element=elm_setup,
                                                                               gdf=x_gdf_line,
                                                                               bool_Guidance=self.bool_Guidance,
                                                                               bool_ABCurve=True,
                                                                               bool_ABLine=True,
                                                                               bool_Adaptive=True,
                                                                               lat=lat,
                                                                               long=long,
                                                                               dict_line_types=self.dict_line_types,
                                                                               line_type_column=self.line_type_column)

            # Save xml
            save_element_xml(root, self.master_data_path)
            # Process int
            value_process = 20 + ((79 * i) / i_max)
            if self.processBar:
                self.processBar.setValue(int(value_process))

        # Input
        elm_setup = create_element.create_inputs(element=elm_setup)

        # Create Chemical
        elm_setup = create_element.create_chemical(element=elm_setup)

        # Create Fertilizer
        elm_setup = create_element.create_fertilizer(element=elm_setup)

        # Create WorkDescriptor
        elm_setup = create_element.create_work_descriptor(element=elm_setup)

        # Create MachineType
        elm_setup = create_element.create_machine_type(element=elm_setup)

        # Create MachineModel
        elm_setup = create_element.create_machine_model(element=elm_setup)

        # Create Machine
        elm_setup = create_element.create_machine(element=elm_setup)

        # Create FlagCategory
        elm_setup = create_element.create_flag_category(element=elm_setup)

        # Create UserDefinedType
        elm_setup = create_element.create_user_define_type(element=elm_setup)

        # Source information
        create_element.create_source_info(elm_setup)

        # Save xml
        save_element_xml(root, self.master_data_path)

        zip_name = os.path.join(self.fold_jd_data, 'Gen4.zip')
        zip_folder(self.fold_gen4, zip_name)

        return self.fold_gen4, self.output_log

    def create_geojson_boundary(self, path_boundary, geom):
        """Create geojson in folder SpatialFiles with Boundary, AdaptiveCurve, ABCurve"""

        self.output_log.append(my_print(f'{now_time(john_deere=False)} - Create boundary'))

        x_geom_dict = {
            'geometry': {
                'coordinates': None,
                'type': 'Polygon'
            },
            'id': 0,
            'properties': {
                'boundarytype': 'Exterior',
                'isactive': True,
                'ispassable': False,
                'name': 'Polygon',
                'offsetid': None,
                'parent': None,
                'signaltype': 0
            },
            'type': 'Feature'
        }

        list_dict_geom = []
        if 'Polygon' == geom.geom_type:
            polygon_ext = list(geom.exterior.coords)
            polygon_int = [list(x.coords) for x in geom.interiors]
            # Exterior polygon
            x_geom_dict['geometry']['coordinates'] = [polygon_ext]
            x_geom_dict['properties']['boundarytype'] = 'Exterior'
            x_geom_dict['properties']['name'] = f'Polygon_{0}'
            x_geom_dict['id'] = 0

            list_dict_geom.append(x_geom_dict.copy())
            list_dict_geom[-1]['geometry'] = x_geom_dict['geometry'].copy()
            list_dict_geom[-1]['properties'] = x_geom_dict['properties'].copy()

            # Check polygon interior is existed
            if len(polygon_int):
                for z, x_inter in enumerate(polygon_int):
                    x_geom_dict['geometry']['coordinates'] = [x_inter]
                    x_geom_dict['properties']['boundarytype'] = 'Interior'
                    x_geom_dict['properties']['name'] = f'Polygon_{z + 1}'
                    x_geom_dict['id'] = z + 1

                    list_dict_geom.append(x_geom_dict.copy())
                    list_dict_geom[-1]['geometry'] = x_geom_dict['geometry'].copy()
                    list_dict_geom[-1]['properties'] = x_geom_dict['properties'].copy()

        else:
            r = 0
            for i, x_polygon in enumerate(geom.geoms):

                polygon_ext = list(x_polygon.exterior.coords)
                polygon_int = [list(x.coords) for x in x_polygon.interiors]

                # Exterior polygon
                x_geom_dict['geometry']['coordinates'] = [polygon_ext]
                x_geom_dict['properties']['boundarytype'] = 'Exterior'
                x_geom_dict['properties']['name'] = f'Polygon_{r}'
                x_geom_dict['id'] = r

                list_dict_geom.append(x_geom_dict.copy())
                list_dict_geom[-1]['geometry'] = x_geom_dict['geometry'].copy()
                list_dict_geom[-1]['properties'] = x_geom_dict['properties'].copy()

                r += 1

                if len(polygon_int):
                    for z, x_inter in enumerate(polygon_int):
                        x_geom_dict['geometry']['coordinates'] = [x_inter]
                        x_geom_dict['properties']['boundarytype'] = 'Interior'
                        x_geom_dict['properties']['name'] = f'Polygon_{r}'
                        x_geom_dict['id'] = r

                        list_dict_geom.append(x_geom_dict.copy())
                        list_dict_geom[-1]['geometry'] = x_geom_dict['geometry'].copy()
                        list_dict_geom[-1]['properties'] = x_geom_dict['properties'].copy()

                        r += 1

        dict_final = {
            'features': list_dict_geom,
            'type': 'FeatureCollection'
        }

        with open(os.path.join(self.fold_spatial, path_boundary), 'w', encoding='UTF-8') as j:
            json.dump(dict_final, j, indent=4)

    def create_geojson_adaptive_curve(self, path_adaptive_curve, gdf_line):
        """Create Adaptive curve"""

        list_multi_lines = []
        for v, x_line in enumerate(gdf_line.geometry.to_list()):
            if x_line.geom_type == 'LineString':
                x_line = [list(x) + [-7000000, 1] for x in list(zip(x_line.xy[0], x_line.xy[1]))]
                list_multi_lines.append(x_line)
            elif x_line.geom_type == 'MultiLineString':
                for i, y_line in enumerate(x_line.geoms):
                    y_line = [list(x) + [-7000000, 1] for x in list(zip(y_line.xy[0], y_line.xy[1]))]
                    list_multi_lines.append(y_line)

        dict_final = {
            "geometry": {
                "coordinates": list_multi_lines,
                "type": "MultiLineString"},
            "type": "Feature"
        }

        with open(os.path.join(self.fold_spatial, path_adaptive_curve), 'w', encoding='UTF-8') as j:
            json.dump(dict_final, j, indent=4)

    def create_geojson_ab_curve(self, path_ab_curve, gdf_line):
        """Create AB curve"""

        list_multi_lines = []
        for v, x_line in enumerate(gdf_line.geometry.to_list()):
            if x_line.geom_type == 'LineString':
                x_line = [list(x) + [0, 1] for x in list(zip(x_line.xy[0], x_line.xy[1]))]
                list_multi_lines.append(x_line)
            elif x_line.geom_type == 'MultiLineString':
                for i, y_line in enumerate(x_line.geoms):
                    y_line = [list(x) + [0, 1] for x in list(zip(y_line.xy[0], y_line.xy[1]))]
                    list_multi_lines.append(y_line)

        # Import line to json
        list_json = []
        for z_line in list_multi_lines:
            list_json.append({
                "geometry": {
                    "coordinates": [z_line],
                    "type": "MultiLineString"},
                "properties": {
                    "curvetype": "Track0"},  # "RightProjections" "LeftProjections"
                "type": "Feature"})

        dict_final = {
            "features": list_json,
            "type": "FeatureCollection"
        }

        with open(os.path.join(self.fold_spatial, path_ab_curve), 'w', encoding='UTF-8') as j:
            json.dump(dict_final, j, indent=4)


class CreateElement(Gen4FromShp):

    def __init__(self, json_input, metadata_element, client='', field='', farm='', crop='', id_process=''):
        super().__init__(json_input=json_input, output_log=[])

        self.client = client
        self.id_client = unique_id(client + id_process)
        self.farm = farm
        self.id_farm = unique_id(farm + id_process)
        self.field = field
        self.id_field = unique_id(field + id_process)
        self.crop = crop
        self.id_crop = unique_id(crop + id_process)
        self.source_node = unique_id('my_jd_farm')
        self.metadata_element = metadata_element
        self.time_now = now_time()
        self.id_process = id_process

    def list_element_parse(self, name_element=''):
        """Parsing element to list of elements"""
        xml_tree = ET.parse(self.metadata_element[name_element])
        root = xml_tree.getroot()
        element_list = [x for x in root.findall(name_element)]

        return element_list

    def create_client(self, element: object) -> object:
        """Client"""
        element_client = self.list_element_parse('Client')[0]
        element_client.attrib['SourceNode'] = self.source_node
        element_client.attrib['StringGuid'] = self.id_client
        element_client.attrib['Name'] = self.client
        element_client.attrib['CreationDate'] = self.time_now
        element_client.attrib['LastModifiedDate'] = self.time_now
        element.append(element_client)

        return element

    def create_farm(self, element: object) -> object:
        """Farm"""
        element_farm = self.list_element_parse('Farm')[0]
        element_farm.attrib['SourceNode'] = self.source_node
        element_farm.attrib['StringGuid'] = self.id_farm
        element_farm.attrib['Client'] = self.id_client
        element_farm.attrib['Name'] = self.farm
        element_farm.attrib['CreationDate'] = self.time_now
        element_farm.attrib['LastModifiedDate'] = self.time_now
        element.append(element_farm)

        return element

    def create_field(self, element: object) -> object:
        """Field"""
        element_field = self.list_element_parse('Field')[0]
        element_field.attrib['SourceNode'] = self.source_node
        element_field.attrib['StringGuid'] = self.id_field
        element_field.attrib['SourceSystemClientId'] = self.id_field
        element_field.attrib['Name'] = self.field
        element_field.attrib['CreationDate'] = self.time_now
        element_field.attrib['LastModifiedDate'] = self.time_now
        element_farm = [x for x in element_field.findall('Farm')][0]
        element_farm.text = self.id_farm

        element.append(element_field)

        return element

    def create_inputs(self, element: object) -> object:
        """Inputs"""
        element_list = self.list_element_parse('Inputs')[0]
        for x_element in element_list:
            element.append(x_element)

        return element

    def create_crop(self, element: object) -> object:
        """Crop"""
        element_crop = self.list_element_parse('Crop')[0]
        element_crop.attrib['SourceNode'] = self.source_node
        element_crop.attrib['Name'] = self.crop
        element_crop.attrib['CreationDate'] = self.time_now
        element_crop.attrib['LastModifiedDate'] = self.time_now
        element.append(element_crop)

        return element

    def create_chemical(self, element: object) -> object:
        """ChemicalType"""
        element_list = self.list_element_parse('ChemicalType')
        for x_element in element_list:
            x_element.attrib['SourceNode'] = self.source_node
            element.append(x_element)

        return element

    def create_fertilizer(self, element: object) -> object:
        """FertilizerType"""
        element_list = self.list_element_parse('FertilizerType')
        for x_element in element_list:
            x_element.attrib['SourceNode'] = self.source_node
            element.append(x_element)

        return element

    def create_work_descriptor(self, element: object) -> object:
        """WorkDescriptor"""
        element_list = self.list_element_parse('WorkDescriptor')
        for x_element in element_list:
            x_element.attrib['SourceNode'] = self.source_node
            element.append(x_element)

        return element

    def create_machine_type(self, element: object) -> object:
        """MachineType"""
        element_list = self.list_element_parse('MachineType')
        for x_element in element_list:
            x_element.attrib['SourceNode'] = self.source_node
            element.append(x_element)

        return element

    def create_machine_model(self, element: object) -> object:
        """MachineModel"""
        element_list = self.list_element_parse('MachineModel')
        for x_element in element_list:
            x_element.attrib['SourceNode'] = self.source_node
            element.append(x_element)

        return element

    def create_machine(self, element: object) -> object:
        """Machine"""
        element_list = self.list_element_parse('Machine')
        for x_element in element_list:
            x_element.attrib['SourceNode'] = self.source_node
            element.append(x_element)

        return element

    def create_flag_category(self, element: object) -> object:
        """FlagCategory"""
        element_list = self.list_element_parse('FlagCategory')
        for x_element in element_list:
            x_element.attrib['SourceNode'] = self.source_node
            x_element.attrib['SourceSystemClientId'] = self.id_field
            element.append(x_element)

        return element

    def create_guidance(self, element: object,
                        gdf: gpd.GeoDataFrame,
                        bool_Guidance: bool,
                        bool_ABLine: bool = True,
                        bool_ABCurve: bool = True,
                        bool_Adaptive: bool = True,
                        lat: float = 49.0,
                        long: float = 16.0,
                        dict_line_types=None,
                        line_type_column: str = 'Type_line') -> object:

        """Guidance"""
        if dict_line_types is None:
            dict_line_types = {"AB Line": 1, "AB Curve": 2, "Adaptive Curve": 3}

        dict_output = {}
        element_guidance = self.list_element_parse('Guidance')[0]
        element_tracks = [x for x in element_guidance.findall('Tracks')][0]
        gdf = gdf_multi_single_line(gdf)
        gdf_ab = gdf[(gdf[line_type_column] == dict_line_types["AB Line"]) & (gdf['count_point'] == 2)]
        gdf_ab_curve = gdf[gdf[line_type_column] == dict_line_types["AB Curve"]]
        gdf_adaptive = gdf[(gdf[line_type_column] == dict_line_types["Adaptive Curve"]) |
                           ((gdf[line_type_column] == dict_line_types["AB Line"]) &
                            (gdf['count_point'] > 2))]

        # Adaptive Curve
        if bool_Adaptive and not gdf_adaptive.empty:

            name_adaptive = f'Adapt-{diacritic(gdf_adaptive["ID_line"].iloc[0])}'
            string_gui = unique_id(name_adaptive + self.id_process)
            path_adaptive = 'AdaptiveCurve' + string_gui + '.gjson'

            element_adaptive = [x for x in element_tracks.findall('AdaptiveCurve')][0]
            element_adaptive.attrib['SourceNode'] = self.source_node
            element_adaptive.attrib['Name'] = name_adaptive
            element_adaptive.attrib['TaggedEntity'] = self.id_field
            element_adaptive.attrib['SourceSystemClientId'] = self.id_field
            element_adaptive.attrib['StringGuid'] = string_gui
            element_adaptive.attrib['CreationDate'] = self.time_now
            element_adaptive.attrib['LastModifiedDate'] = self.time_now

            # Geometry
            element_geometry = [x for x in element_adaptive.findall('Geometry')][0]
            element_with_extension = [x for x in element_geometry.findall('FilenameWithExtension')][0]
            element_with_extension.text = path_adaptive

            # ReferenceLatitude + ReferenceLongitude
            element_reference_lat = [x for x in element_adaptive.findall('ReferenceLatitude')][0]
            element_reference_lat.attrib['Value'] = str(lat)  # ????
            element_reference_long = [x for x in element_adaptive.findall('ReferenceLongitude')][0]
            element_reference_long.attrib['Value'] = str(long)  # ????
            dict_output['Adaptive'] = path_adaptive

            # If create Guidance element
            if bool_Guidance:
                element_guidance_own = ET.Element('Guidance')
                element_tracks_own = ET.SubElement(element_guidance_own, 'Tracks')
                element_tracks_own.append(element_adaptive)
                element.append(element_guidance_own)
                bool_Guidance = False
            else:
                element_guidance_own = [x for x in element.findall('Guidance')][0]
                element_tracks_own = [x for x in element_guidance_own.findall('Tracks')][0]
                element_tracks_own.append(element_adaptive)

            # Create Adaptive Line to json
            super().create_geojson_adaptive_curve(dict_output['Adaptive'], gdf_adaptive)

        # AB Line
        if bool_ABLine and not gdf_ab.empty:

            # Count point
            first_step = True
            for i in range(gdf_ab.shape[0]):
                name_ab_line = f'AB-{diacritic(gdf_ab["ID_line"].iloc[i])}'
                string_gui = unique_id(name_ab_line + self.id_process)

                element_ab_line = [x for x in element_tracks.findall('ABLine')][0]
                element_ab_line.attrib['SourceNode'] = self.source_node
                element_ab_line.attrib['Name'] = name_ab_line
                element_ab_line.attrib['TaggedEntity'] = self.id_field
                element_ab_line.attrib['SourceSystemClientId'] = self.id_field
                element_ab_line.attrib['StringGuid'] = string_gui
                element_ab_line.attrib['CreationDate'] = self.time_now
                element_ab_line.attrib['LastModifiedDate'] = self.time_now

                # APoint + BPoint
                element_a_point = [x for x in element_ab_line.findall('APoint')][0]
                element_a_point.attrib['Latitude'] = str(np.array(gdf_ab.geometry.iloc[i].coords)[0][1])
                element_a_point.attrib['Longitude'] = str(np.array(gdf_ab.geometry.iloc[i].coords)[0][0])
                element_b_point = [x for x in element_ab_line.findall('BPoint')][0]
                element_b_point.attrib['Latitude'] = str(np.array(gdf_ab.geometry.iloc[i].coords)[-1][1])
                element_b_point.attrib['Longitude'] = str(np.array(gdf_ab.geometry.iloc[i].coords)[-1][0])
                element_heading = [x for x in element_ab_line.findall('Heading')][0]
                element_heading.attrib['Value'] = str(gdf_ab['length'].iloc[i])

                # If create Guidance element
                if bool_Guidance:
                    element_guidance_own = ET.Element('Guidance')
                    element_tracks_own = ET.SubElement(element_guidance_own, 'Tracks')
                    element_tracks_own.append(deepcopy(element_ab_line))
                    element.append(deepcopy(element_guidance_own))
                    bool_Guidance = False
                else:
                    if first_step:
                        element_guidance_own = [x for x in element.findall('Guidance')][0]
                        element_tracks_own = [x for x in element_guidance_own.findall('Tracks')][0]
                        first_step = False
                    element_tracks_own.append(deepcopy(element_ab_line))  # Very important for copy in memory!

        # AB Curve
        if bool_ABCurve and not gdf_ab_curve.empty:

            # Count point
            first_step = True
            for i in range(gdf_ab_curve.shape[0]):

                name_ab_curve = f'ABCurve-{diacritic(gdf_ab_curve["ID_line"].iloc[i])}'
                string_gui = unique_id(name_ab_curve + self.id_process)
                path_adaptive = 'ABCurve' + string_gui + '.gjson'

                element_ab_curve = [x for x in element_tracks.findall('ABCurve')][0]
                element_ab_curve.attrib['SourceNode'] = self.source_node
                element_ab_curve.attrib['Name'] = name_ab_curve
                element_ab_curve.attrib['TaggedEntity'] = self.id_field
                element_ab_curve.attrib['SourceSystemClientId'] = self.id_field
                element_ab_curve.attrib['StringGuid'] = string_gui
                element_ab_curve.attrib['CreationDate'] = self.time_now
                element_ab_curve.attrib['LastModifiedDate'] = self.time_now

                # Geometry
                element_geometry = [x for x in element_ab_curve.findall('Geometry')][0]
                element_with_extension = [x for x in element_geometry.findall('FilenameWithExtension')][0]
                element_with_extension.text = path_adaptive

                # APoint + BPoint
                element_a_point = [x for x in element_ab_curve.findall('APoint')][0]
                element_a_point.attrib['Latitude'] = str(np.array(gdf_ab_curve.geometry.iloc[i].coords)[0][1])
                element_a_point.attrib['Longitude'] = str(np.array(gdf_ab_curve.geometry.iloc[i].coords)[0][0])
                element_b_point = [x for x in element_ab_curve.findall('BPoint')][0]
                element_b_point.attrib['Latitude'] = str(np.array(gdf_ab_curve.geometry.iloc[i].coords)[-1][1])
                element_b_point.attrib['Longitude'] = str(np.array(gdf_ab_curve.geometry.iloc[i].coords)[-1][0])
                element_heading = [x for x in element_ab_curve.findall('Heading')][0]
                element_heading.attrib['Value'] = str(gdf_ab_curve['length'].iloc[i])
                dict_output['ABCurve'] = path_adaptive

                # If create Guidance element
                if bool_Guidance:
                    element_guidance_own = ET.Element('Guidance')
                    element_tracks_own = ET.SubElement(element_guidance_own, 'Tracks')
                    element_tracks_own.append(element_ab_curve)
                    element.append(element_guidance_own)
                    bool_Guidance = False
                else:
                    if first_step:
                        element_guidance_own = [x for x in element.findall('Guidance')][0]
                        element_tracks_own = [x for x in element_guidance_own.findall('Tracks')][0]
                        first_step = False

                    element_tracks_own.append(deepcopy(element_ab_curve))

                    # Create Adaptive Line to json
                super().create_geojson_ab_curve(dict_output['ABCurve'], gdf_ab_curve)

        else:
            pass

        return element, bool_Guidance

    def create_boundary(self, element: object, idn: int) -> object:
        """OperationalBoundary"""
        name_boundary = self.field + str(idn) + self.id_process
        string_gui = unique_id(name_boundary)
        path_boundary = 'Boundary' + string_gui + '.gjson'

        element_boundary = self.list_element_parse('OperationalBoundary')[0]
        element_boundary.attrib['SourceNode'] = self.source_node
        element_boundary.attrib['SourceSystemClientId'] = self.id_field  # ??? field + FlagCategory
        element_boundary.attrib['StringGuid'] = string_gui
        element_boundary.attrib['TaggedEntity'] = self.id_field
        element_boundary.attrib['Name'] = name_boundary
        element_boundary.attrib['CreationDate'] = self.time_now
        element_boundary.attrib['LastModifiedDate'] = self.time_now

        element_geometry = [x for x in element_boundary.findall('Geometry')][0]
        element_file_name_ext = [x for x in element_geometry.findall('FilenameWithExtension')][0]
        element_file_name_ext.text = path_boundary

        element.append(element_boundary)

        return element, path_boundary

    def create_user_define_type(self, element: object) -> object:
        """UserDefinedType"""
        element_list = self.list_element_parse('UserDefinedType')
        for x_element in element_list:
            x_element.attrib['SourceNode'] = self.source_node
            element.append(x_element)

        return element

    def create_source_info(self, element: object) -> object:
        """SourceInformation"""
        element_list = self.list_element_parse('SourceInformation')
        for x_element in element_list:
            element.append(x_element)

        return element


def create_file(input_json, processBar=None):
    # Logging and turn of
    current_path = os.path.join(input_json['path_home'], 'log_gen4.log')
    logging.basicConfig(filename=current_path, level='INFO')
    logger_blocklist = [
        "fiona",
        "geopandas",
        "gdal",
        "rasterio",
        "matplotlib",
        "PIL"]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    logging.info("====================")
    logging.info('Cas: {} - START'.format(f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S%z}'))

    # Check if shapefile is existed
    if not os.path.exists(input_json["shpFields"]):
        path_shp = input_json["shpFields"]
        output_folder = ""
        output_log = [f"""<a>{now_time(john_deere=False)} - The path to the Shapefile with polygons does not exist:</a>""",
                      f"""<a style="color: blue; text-decoration: underline;">{path_shp}</a>"""]

        return output_log, output_folder

    if not os.path.exists(input_json["shpGuide"]) and (len(input_json["shpGuide"]) > 0):
        path_shp = input_json["shpGuide"]
        output_folder = ""
        output_log = [f"""<a>{now_time(john_deere=False)} - The path to the Shapefile with polylines does not exist:</a>""",
                      f"""<a style="color: blue; text-decoration: underline;">{path_shp}</a>"""]

        return output_log, output_folder

    # Remove JD Data
    fold_jd_data = os.path.dirname(input_json["shpFields"])
    fold_jd_data = os.path.join(fold_jd_data, "JD data")
    try:
        if os.path.isdir(fold_jd_data):
            shutil.rmtree(fold_jd_data)

        input_json['json_metadat'] = os.path.join(input_json['path_home'], input_json['json_metadat'])
        gen4_from_shp = Gen4FromShp(input_json, output_log=[], processBar=processBar)
        gen4_from_shp.prepare_gdf()
        output_folder, output_log = gen4_from_shp.creat_setup_xml()

    except PermissionError:
        output_folder = ""
        output_log = [my_print(f"The process cannot access the file in the JD data folder because "
                               f"it is currently being used by another process!")]
    except Exception as e:
        output_folder = ""
        output_log = [my_print(
            f"{now_time(john_deere=False)} - {type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")]
        logging.exception("An exception was thrown!")
    finally:
        logging.info('Cas: {} - STOP'.format(f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S%z}'))

    return output_log, output_folder


if __name__ == '__main__':
    dict_test = {
    "path_home": "C:/Users/Petr/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/qgis-convertshpgen4-plugin",
    "json_metadat": "gen4_scripts/base_element/version1/metadata.json",
    "shpFields": "C:/Users/Petr/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/qgis-convertshpgen4-plugin/data/boundary_polygon.shp",
    "shpGuide": "C:/Users/Petr/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/qgis-convertshpgen4-plugin/data/navigation_line.shp",
    "clientName": "ROSTENICE 2023",
    "farmColumn": "STREDISKO",
    "fieldColumn": "PARCELA",
    "cropColumn": "",
    "lineColumn": "ID_str",
    "lineColumnType": "Line_type"
}
    create_file(dict_test)
