import multiprocessing
import os, logging
import PIL
import json
import numpy as np
from PIL import Image as I
from shapely.geometry import Point, Polygon, asMultiPoint
import cv2
import pickle
import math
from scipy import stats
from tqdm import tqdm
import matplotlib.path as mpltPath

# ====================Read Images and Labels====================

Image_List = []
Image_Dir = None
# Label Dict format - {image_filename:[Polygon Tuples]}
Label = {}


# Read and Verify the Images
# Input:    Image Directory Path
# Output:   Image Directory Path, Image Lists
def readImages(path):
    global Image_List, Image_Dir
    Image_Dir = os.path.abspath(path)
    Image_List = []
    for file in os.listdir(Image_Dir):
        i = os.path.join(Image_Dir, file)
        if os.path.isdir(i):
            if not i.endswith(".images_temp"):
                logging.warning("Found a directory ({0}). MLMonkey has ignored it.".format(i))
            continue
        try:
            I.open(i)
            Image_List.append(file)
        except PIL.UnidentifiedImageError:
            logging.warning("{0} cannot be opened and MLMonkey has ignored it.".format(i))

    return Image_Dir, Image_List


# Add Polygon into MLMonkey
# Input:    Image Filename
#           Polygon Tuples
# Output:   Current Label Dictionary
def addLabel(image_filename, polygon):
    global Label
    if image_filename not in Label.keys():
        Label[image_filename] = []
    Label[image_filename].append(polygon)
    return Label


# Initialise the Label Dictionary
def initLabel():
    global Label
    Label = {}


# Convert the Split Polygon X and Y to a list of Tuple
# Input:    X coordinates - List or Numpy
#           Y coordinate - List or Numpy
# Output:   Numpy List
def convertPolytoTuple(x, y):
    if len(x) != len(y):
        raise ValueError(
            "The X and Y coordinates do not have same length (X has got {0} & Y has got {1})".format(len(x), len(y)))
    return np.array([(x[i], y[i]) for i in range(len(x))])


# Load VIA Labels
# Input:    JSON File Path
# Output:   Label Dictionary
def readVIALabel(file, init=False, test_gen=False):
    global Label
    if init:
        Label = {}
    with open(os.path.abspath(file)) as f:
        json_label = json.loads(f.read())
        f.close()
    for i in json_label.values():
        if test_gen:
            if i['filename'].split("/")[-1] in Image_List:
                for j in i['regions']:
                    addLabel(i['filename'].split("/")[-1], convertPolytoTuple(j['shape_attributes']['all_points_x'],
                                                                              j['shape_attributes']['all_points_y']))
        else:
            if i['filename'] in Image_List:
                for j in i['regions']:
                    addLabel(i['filename'], convertPolytoTuple(j['shape_attributes']['all_points_x'],
                                                               j['shape_attributes']['all_points_y']))
    return Label


# Check data is correctly loaded
# Process: Logging the error
def check_data():
    global Label
    if len(Label.keys()) <= 0:
        logging.error("Label is not correctly loaded")


# ====================Preprocessing====================

Label_ = {}


# Calculate Bounding Box Area of a polygon in images size
# Input:    List of Tuples - Polygon
# Output:   (Left, Top, Right, Bottom)
def cal_boundary(polygon):
    min_x = min([i[0] for i in polygon])
    min_y = min([i[1] for i in polygon])
    max_x = max([i[0] for i in polygon])
    max_y = max([i[1] for i in polygon])
    return [min_x, min_y, max_x, max_y]


# Crop the defect area and stored into a temp file
# Store the cropped polygon data into dictionary
# Input:    Image ID
#           Crop Padding - Unit: pixels
def crop(did, pad):
    global Label_
    img = I.open(os.path.join(Image_Dir, Label_[did]['filename']))
    boundary = cal_boundary(Label_[did]['polygon'])
    Label_[did]["boundary"] = (boundary, [img.width, img.height])
    Label_[did]["width"] = (boundary[3] - boundary[1])
    Label_[did]["height"] = (boundary[2] - boundary[0])
    padX = int((boundary[2] - boundary[0]) * pad)
    padY = int((boundary[3] - boundary[1]) * pad)
    if padX <= 2:
        padX = 5
    if padY <= 2:
        padY = 5
    boundaryF = [0, 0, 0, 0]
    boundaryF[0] = boundary[0] - padX
    boundaryF[1] = boundary[1] - padY
    boundaryF[2] = boundary[2] + padX
    boundaryF[3] = boundary[3] + padY
    offset = [boundary[0] - padX, boundary[1] - padY]
    if boundaryF[0] < 0:
        offset[0] = 0
        boundaryF[0] = 0
    if boundaryF[1] < 0:
        offset[1] = 0
        boundaryF[1] = 0
    if boundaryF[2] >= img.size[0]:
        boundaryF[2] = img.size[0] - 1
    if boundaryF[3] >= img.size[1]:
        boundaryF[3] = img.size[1] - 1
    cropped_img = img.crop(boundaryF)
    cropped_img.save(os.path.join("./.images_temp", str(did)) + ".jpg")
    Label_[did]["crop_poly"] = np.array([(i[0] - offset[0], i[1] - offset[1]) for i in Label_[did]['polygon']])


# Load Image
# Input: Image directory path
# Output: Image array
def loadImage(path):
    img = I.open(path)
    return np.array(img)


# Convert image to map, which determine each pixel is inside of polygon or not
# Input: Image shape,
#        Polygon coordinates list
# Output: Map array
def cal_map(shape, poly):
    p = np.array([[i[1], i[0]] for i in poly])

    # Method 1: 4-5s/img
    # poly = Polygon(p)
    # poly_map = np.array([[poly.contains(Point((w, h))) for w in range(shape[1])] for h in range(shape[0])]).astype(bool)

    # Method 2 : 3-4s/img
    # poly = Polygon(p)
    # w, h = np.meshgrid(range(shape[1]), range(shape[0]))
    # coor_l = list(np.dstack((h, w)).reshape((shape[0] * shape[1], 2)))
    # poly_map = np.array(list(map(lambda x: poly.contains(Point(x)), coor_l))).reshape((shape[0], shape[1]))

    # Method 3: 2.5-3.5s/img
    # poly = Polygon(p)
    # w, h = np.meshgrid(range(shape[1]), range(shape[0]))
    # coor = np.dstack((h, w)).reshape((shape[0] * shape[1], 2))
    # poly_map = np.array(list(map(poly.contains, list(map(Point, coor))))).reshape((shape[0], shape[1]))

    # Method 4: 0.18-0.2s/img
    poly = mpltPath.Path(p)
    w, h = np.meshgrid(range(shape[1]), range(shape[0]))
    coor = np.dstack((h, w)).reshape((shape[0] * shape[1], 2))
    poly_map = poly.contains_points(coor).reshape((shape[0], shape[1]))

    return poly_map


# Convert RGB to different colour mode
# Input: Image array, mode=(HSV, HLS, LAB)
# Output: Image array
def RGBConvert(img_array, mode="HSV"):
    if mode == "HSV" or mode == "HSB":
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV_FULL)
    elif mode == "HLS":
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2HLS_FULL)
    elif mode == "LAB":
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    else:
        raise ValueError("Non Supported Mode: {0}".format(mode))


# Calculate bounding box size
# Input: Polygon coordinate list:List
# Output: Bounding box area:Integer
def cal_bb_size(poly):
    boundary = cal_boundary(poly)
    return int((boundary[2] - boundary[0]) * (boundary[3] - boundary[1]))


# Calculate polygon size
# Input: Polygon map: Array
# Output: Polygon size: Integer
def cal_poly_size(poly_map):
    return np.count_nonzero(poly_map)


# Calculate distance between two polygons
# Input: Polygon 1: List
#        Polygon 2: List
# Output: Distance: Integer
def cal_distance(poly1, poly2):
    return Polygon(poly1).distance(Polygon(poly2))


# Find the closest neighbour
# Input: Image_ID
# Output: Neighbour List: List(Image_ID)
def find_neighbour(img_id):
    return list(filter(lambda x: (int(x) == int(img_id)) and (x != img_id), list(Label_.keys())))


# Calculate shape complexity
# Input: Polygon coordinate list
# Output: Angle List:Array(Angle),
#         Number of Edge:Integer,
#         Bounding length list:List(Length)
def cal_shape(poly):
    polygon = np.append([poly[-1]], poly, axis=0)
    polygon = np.append(polygon, [poly[0]], axis=0)
    edge = 0
    degree = []
    length = []
    turning = []
    for i in range(1, len(polygon) - 1):
        c = math.dist(polygon[i - 1], polygon[i + 1])
        a = math.dist(polygon[i - 1], polygon[i])
        b = math.dist(polygon[i], polygon[i + 1])
        if a <= 0 or b <= 0:
            continue
        length.append((a, b))
        angle = math.degrees(math.acos(round((a ** 2 + b ** 2 - c ** 2) / (2 * a * b), 5)))
        degree.append(round(angle))
        goin_yd = polygon[i][-1] - polygon[i - 1][-1]
        goin_xd = polygon[i][0] - polygon[i - 1][0]
        if goin_xd == 0:
            # vertical direction
            if goin_yd >= 0:
                # going down
                if polygon[i + 1][0] <= polygon[i][0]:
                    # turn right
                    turning.append(180 - angle)
                else:
                    # turn left
                    turning.append(-180 + angle)
            else:
                # going up
                if polygon[i + 1][0] < polygon[i][0]:
                    # turn left
                    turning.append(-180 + angle)
                else:
                    # turn right
                    turning.append(180 - angle)
        else:
            # non vertical direction
            goin_a = goin_yd / goin_xd
            goin_b = polygon[i][-1] - goin_a * polygon[i][0]
            inter_y = goin_a * polygon[i + 1][0] + goin_b
            if goin_xd >= 0:
                # going right
                if polygon[i + 1][-1] >= inter_y:
                    # turn right
                    turning.append(180 - angle)
                else:
                    # turn left
                    turning.append(-180 + angle)
            else:
                # going left
                if polygon[i + 1][-1] > inter_y:
                    # turn left
                    turning.append(-180 + angle)
                else:
                    # turn right
                    turning.append(180 - angle)
        if angle < 170:
            edge += 1
    turning = np.array(turning).astype(int)
    return np.array(degree), edge, np.array(length), np.array(turning)


# Convert the Labels to ID based Data
# Input:    Colour Mode - Default: HSV/HSB
#           crop_pad (Optional) - Default: 0 - crop extra outside area with percentage
#           save (Optional) - If you want to save the loaded data, please overwrite the saving directory
#           reload (Optional) - Loading the stored data
# Output:   Label_reID Dictionary - Result is stored in Label_reID
def loadData(color_mode="HSV", crop_pad=0, save=None, reload=None):
    global Label_, Label
    check_data()
    Label_ = {}
    count = 1.0
    if reload is not None:
        with open(os.path.abspath(reload), 'rb') as f:
            Label_ = pickle.load(f)
            f.close()

        return Label_

    for i in tqdm(Label.keys(), desc="Loading Images"):
        count2 = 0.1
        for j in Label[i]:
            did = count + count2
            Label_[did] = {'filename': i, 'iid': int(count), 'did': int(10 * count2), 'polygon': j}
            crop(did, crop_pad)
            img_array = loadImage(os.path.join("./.images_temp", str(did) + ".jpg"))
            Label_[did]["img_arr"] = img_array
            Label_[did]["map"] = cal_map(Label_[did]["img_arr"].shape, Label_[did]["crop_poly"])
            Label_[did]["img_arr_mode"] = RGBConvert(Label_[did]["img_arr"], mode=color_mode)
            Label_[did]["bb_size"] = cal_bb_size(Label_[did]["crop_poly"])
            Label_[did]["poly_size"] = cal_poly_size(Label_[did]["map"])
            Label_[did]["hue"] = Label_[did]["img_arr_mode"][:, :, 0]
            Label_[did]["sat"] = Label_[did]["img_arr_mode"][:, :, 1]
            Label_[did]["value"] = Label_[did]["img_arr_mode"][:, :, 2]
            Label_[did]["neighbour_dist"] = [cal_distance(Label_[did]["polygon"], Label_[i]["polygon"]) for i in
                                             find_neighbour(did)]
            Label_[did]["degree"], Label_[did]["edge"], Label_[did]["edge_len"], Label_[did]["turning"] = \
                cal_shape(Label_[did]["polygon"])

            count2 += 0.1

        count += 1.0

    if save is not None:
        with open(os.path.abspath(save), 'wb') as f:
            pickle.dump(Label_, f)
            f.close()

    return Label_


# ====================Feature Extraction====================


feature_list = []


# Split values into group
# Input: Value,
#        Group_boundary
# Output: Grouped_value:Integer
def cal_group(value, group):
    if value <= group[0]:
        return 1
    elif group[0] < value <= group[1]:
        return 2
    elif group[1] < value <= group[2]:
        return 3
    elif group[2] < value <= group[3]:
        return 4
    else:
        return 5


# Calculate group boundary
# Input: Array:Array(value),
#        Boundary gap:Float
# Output: Group list:List
def group(arr, gap=0.30):
    s = sorted(arr)
    l = [int(len(s) * i / 5) for i in range(1, 5)]
    group = []
    for i in l:
        index = (i, 0)
        for j in range(i - round(gap * l[0] / 2), i + round(gap * l[0] / 2)):
            if s[j + 1] - s[j] >= index[-1]:
                index = (j, s[j + 1] - s[j])
        group.append(index[0])
    return [s[i] for i in group]


# Group Number of Edge
# Input: Edge_List
# Output: Average edge number:Integer
#         Mode edge number:Integer
def cal_edge(edge):
    ratio = []
    for i in edge:
        ratio.append(round(min([i[0] / i[1], i[1] / i[0]]) * 4) + 1)

    ratio = np.array(ratio)
    return round(np.average(ratio)), int(stats.mode(ratio, axis=None)[0])


# Group the Angles
# Input: Angle_List
# Output: Average angle:Integer
#         Mode angle:Integer
def cal_deg(deg):
    d = (deg + 30) / 30
    d = d.astype(int)
    avg_d = int((np.average(deg) + 30) / 30)
    if avg_d > 6:
        avg_d = 6

    return avg_d, int(stats.mode(d, axis=None)[0])


def cal_shape_comp(turning, deg, edge):
    t = np.append([turning[-1]], turning, axis=0)
    total_len = sum(edge[:, 0])
    edge_per = edge / total_len
    edge_per = np.append([edge_per[-1]], edge_per, axis=0)
    follow_turn = 0
    reverse_turn = 0
    small_turn = 0
    edge_ratio = 0
    sc_score = 0
    for i in range(1, len(t)):

        turn_v = abs(t[i] - t[i - 1]) / 360

        if (t[i] > 0 and t[i - 1] < 0) or (t[i] < 0 and t[i - 1] > 0):
            reverse_turn += 1
            follow_turn -= 1 - turn_v

        else:
            follow_turn += 1 - turn_v
        if abs(t[i]) >= 90:
            small_turn += 1
        er = 1 / (max(edge_per[i]) / min(edge_per[i]))
        edge_ratio += er
        sc_score += (abs(t[i]) / 180) * er

    follow_turn /= len(turning)
    edge_ratio /= len(turning)
    sc_score /= len(turning)
    if follow_turn > 1:
        follow_turn = 1
    if follow_turn < 0:
        follow_turn = 0
    reverse_turn /= len(turning)
    small_turn /= len(turning)

    return round(edge_ratio * 10), round(follow_turn * 10), round(reverse_turn * 10), round(small_turn * 10), round(
        sc_score * 10)


# Calculate and group coverage of polygon in bounding box
# Input: Polygon coordinate list
#        Bounding
# Output: Coverage:Float - the percentage of polygon in bounding box
def cal_coverage(poly, bb):
    cvg = round(((poly / bb) + 0.1) * 5)
    if cvg > 5:
        cvg = 5
    return cvg


# Calculate and group aspect ratio of bounding box
# Input: Polygon
# Output: Aspect Ratio:Integer - Calculate the ratio between long side and short side by long side / short side,
#         then round it into the closest integer
def cal_asp_ratio(poly):
    boundary = cal_boundary(poly)
    ratio = max([(boundary[2] - boundary[0]) / (boundary[3] - boundary[1]),
                 (boundary[3] - boundary[1]) / (boundary[2] - boundary[0])])
    return int(round(ratio, 1) + 0.9)


# Group neighbour distance
# Input: Neighbour distance list
#        Threshold - short distance threshold
# Output: Grouped Distance - 1: short, 2: long, 3: No neighbour
def cal_dist(dist, threshold):
    if threshold == 0:
        if len(dist) == 0:
            return -1
        else:
            return min(dist)
    if len(dist) == 0:
        return 3
    else:
        short = min(dist)
        if short <= threshold:
            return 1
        else:
            return 2


# Group hue values - Hue is degree-based value, 0 = 360, so max range is 180
# Input: Hue Array
#        Outside of polygon:Boolean - group polygon inside or outside of hue
#        hue map:Array- polygon map
# Output: Average Hue:Integer
#         Mode Hue:Integer
#         Hue Range:Integer - range of min and max Hue values
#         Unique Hue:Integer - unique number of Hue
def cal_hue(hue, out, hmap):
    def hue_range(hu):
        hue_r = 0
        hue_set = list(set(hu))
        for i in hue_set:
            max_r = max(list(map(lambda x: min([abs(i - x), abs(i + 12 - x), abs(x + 12 - i)]), hue_set)))
            if max_r > hue_r:
                hue_r = max_r
        return hue_r

    if out:
        h = hue[np.logical_not(hmap)]
    else:
        h = hue[hmap]

    avg_h = np.average(h)
    if avg_h >= 345:
        avg_h = 1
    else:
        avg_h = int((avg_h + 45) / 30)
    h = h.astype(float)
    h = (h + 45) / 30
    h[h >= 13] = 1
    h = h.astype(int)
    distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}
    for v, c in list(zip(stats.find_repeats(h).values, stats.find_repeats(h).counts)):
        distribution[int(v)] = round(c / len(h), 2)

    return avg_h, int(stats.mode(h, axis=None)[0]), int(hue_range(h)), len(np.unique(h)), distribution


# Group saturation or brightness values
# Input: Saturation or brightness Array
#        Outside of polygon:Boolean - group polygon inside or outside of hue
#        hue map:Array- polygon map
# Output: Average Saturation or brightness:Integer
#         Mode Saturation or brightness:Integer
#         Saturation or brightness Range:Integer - range of min and max Saturation or brightness values
#         Unique Saturation or brightness:Integer - unique number of Saturation or brightness
def cal_sat_brt(sat_brt, out, hmap):
    if out:
        sb = sat_brt[np.logical_not(hmap)]
    else:
        sb = sat_brt[hmap]

    avg_sb = np.average(sb)

    if avg_sb >= 255:
        avg_sb = 5
    else:
        avg_sb = int(((avg_sb / 255) + 0.2) * 5)
    sb = sb.astype(float)
    sb = ((sb / 255) + 0.2) * 5
    sb[sb >= 6] = 5
    sb = sb.astype(int)
    distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for v, c in list(zip(stats.find_repeats(sb).values, stats.find_repeats(sb).counts)):
        distribution[int(v)] = round(c / len(sb), 2)

    return avg_sb, int(stats.mode(sb, axis=None)[0]), int(max(sb) - min(sb)), len(np.unique(sb)), distribution


# Colour Complexity (Further work)
def cal_comp_colour(hue_dist, sat_dist, brt_dist, out_hue_dist, out_sat_dist, out_brt_dist):
    hue_outin = 0
    sat_outin = 0
    brt_outin = 0
    for i in hue_dist:
        hue_outin += abs(hue_dist[i] - out_hue_dist[i]) / len(hue_dist)

    for i in sat_dist:
        sat_outin += abs(sat_dist[i] - out_sat_dist[i]) / len(sat_dist)

    for i in brt_dist:
        brt_outin += abs(brt_dist[i] - out_brt_dist[i]) / len(brt_dist)
    hue_outin = round(hue_outin * 10)
    sat_outin = round(sat_outin * 10)
    brt_outin = round(brt_outin * 10)
    return hue_outin, sat_outin, brt_outin


def norm_data(data):
    return {"min": min(data), "max": max(data)}


# Extract feature Function
# Input:    Label_reID
# Output:   List - Store all features
def featureExtract(outside="mode", distance_threshold=100, shape_detail="full", colour_detail="full",
                   save=None, reload=None):
    global Label_, feature_list
    feature_list = []
    if outside is not None:
        if outside == "average":
            feature_list.extend(["out_hue_avg", "out_sat_avg", "out_brt_avg"])
        elif outside == "mode":
            feature_list.extend(["out_hue_mode", "out_sat_mode", "out_brt_mode"])
        elif colour_detail == "unique":
            feature_list.extend(["out_hue_uni", "out_sat_uni", "out_brt_uni"])
        elif outside == "full":
            feature_list.extend(["out_hue_avg", "out_hue_mode", "out_hue_range", "out_hue_uni",
                                 "out_sat_avg", "out_sat_mode", "out_sat_range", "out_sat_uni",
                                 "out_brt_avg", "out_brt_mode", "out_brt_range", "out_brt_uni"])
        else:
            raise AttributeError('Cannot recognise outside attribute. Expect ("mode", "average", "full", None')
    if shape_detail is not None:
        if shape_detail == "basic":
            feature_list.extend(["size", "coverage", "asp_ratio", "deg_avg", "deg_mode", "edge",
                                 "edgelen_avg", "edgelen_mode", "distance"])
        if shape_detail == "complex":
            feature_list.extend(["sc_edge_ratio", "sc_follow_turn", "sc_reverse_turn", "sc_small_turn", "sc_score"])
        if shape_detail == "full":
            feature_list.extend(["size", "coverage", "asp_ratio", "deg_avg", "deg_mode", "edge",
                                 "edgelen_avg", "edgelen_mode", "distance"])
            feature_list.extend(["sc_edge_ratio", "sc_follow_turn", "sc_reverse_turn", "sc_small_turn", "sc_score"])
    else:
        raise AttributeError('Cannot recognise shape detail attribute. Expect ("basic", "complex", "full")')
    if colour_detail is not None:
        if colour_detail == "average":
            feature_list.extend(["hue_avg", "sat_avg", "brt_avg"])
        if colour_detail == "mode":
            feature_list.extend(["hue_mode", "sat_mode", "brt_mode"])
        if colour_detail == "unique":
            feature_list.extend(["hue_uni", "sat_uni", "brt_uni"])
        if colour_detail == "complex":
            feature_list.extend(["hue_outin", "sat_outin", "brt_outin"])
        if colour_detail == "full":
            feature_list.extend(["hue_avg", "hue_mode", "hue_range", "hue_uni",
                                 "sat_avg", "sat_mode", "sat_range", "sat_uni",
                                 "brt_avg", "brt_mode", "brt_range", "brt_uni", "hue_outin", "sat_outin", "brt_outin"])
    else:
        raise AttributeError(
            'Cannot recognise outside attribute. Expect ("unique", "complex", "mode", "average", "full")')
    if reload is not None:
        with open(os.path.abspath(reload), 'rb') as f:
            Label_ = pickle.load(f)
            f.close()
        return Label_
    norm_size = norm_data([int(Label_[i]["poly_size"]) for i in Label_])
    norm_edge = norm_data([int(Label_[i]["edge"]) for i in Label_])
    for did in tqdm(Label_.keys(), desc="Feature Extraction"):
        # Label_[did]["grouped_size"] = cal_group(Label_[did]["poly_size"], size_group)
        Label_[did]["size"] = round(9 * (Label_[did]["poly_size"] - norm_size["min"]) /
                                    (norm_size["max"] - norm_size["min"])) + 1
        Label_[did]["coverage"] = cal_coverage(Label_[did]["poly_size"], Label_[did]["bb_size"])
        Label_[did]["asp_ratio"] = cal_asp_ratio(Label_[did]["crop_poly"])
        Label_[did]["deg_avg"], Label_[did]["deg_mode"] = cal_deg(Label_[did]["degree"])
        Label_[did]["edge"] = round(9 * (Label_[did]["edge"] - norm_edge["min"]) /
                                    (norm_edge["max"] - norm_edge["min"])) + 1
        Label_[did]["edgelen_avg"], Label_[did]["edgelen_mode"] = cal_edge(Label_[did]["edge_len"])
        Label_[did]["sc_edge_ratio"], Label_[did]["sc_follow_turn"], Label_[did]["sc_reverse_turn"], \
        Label_[did]["sc_small_turn"], Label_[did]["sc_score"] = cal_shape_comp(Label_[did]["turning"],
                                                                               Label_[did]["degree"],
                                                                               Label_[did]["edge_len"])
        Label_[did]["distance"] = cal_dist(Label_[did]["neighbour_dist"], distance_threshold)
        Label_[did]["hue_avg"], Label_[did]["hue_mode"], Label_[did]["hue_range"], Label_[did]["hue_uni"], \
        Label_[did]["hue_dist"] = cal_hue(Label_[did]["hue"], False, Label_[did]["map"])
        Label_[did]["sat_avg"], Label_[did]["sat_mode"], Label_[did]["sat_range"], Label_[did]["sat_uni"], \
        Label_[did]["sat_dist"] = cal_sat_brt(Label_[did]["sat"], False, Label_[did]["map"])
        Label_[did]["brt_avg"], Label_[did]["brt_mode"], Label_[did]["brt_range"], Label_[did]["brt_uni"], \
        Label_[did]["brt_dist"] = cal_sat_brt(Label_[did]["value"], False, Label_[did]["map"])

        Label_[did]["out_hue_avg"], Label_[did]["out_hue_mode"], Label_[did]["out_hue_range"], Label_[did][
            "out_hue_uni"], Label_[did]["out_hue_dist"] = cal_hue(Label_[did]["hue"], True, Label_[did]["map"])
        Label_[did]["out_sat_avg"], Label_[did]["out_sat_mode"], Label_[did]["out_sat_range"], Label_[did][
            "out_sat_uni"], Label_[did]["out_sat_dist"] = cal_sat_brt(Label_[did]["sat"], True, Label_[did]["map"])
        Label_[did]["out_brt_avg"], Label_[did]["out_brt_mode"], Label_[did]["out_brt_range"], Label_[did][
            "out_brt_uni"], Label_[did]["out_brt_dist"] = cal_sat_brt(Label_[did]["value"], True, Label_[did]["map"])
        Label_[did]["hue_outin"], Label_[did]["sat_outin"], Label_[did]["brt_outin"] = cal_comp_colour(
            Label_[did]["hue_dist"], Label_[did]["sat_dist"], Label_[did]["brt_dist"], Label_[did]["out_hue_dist"],
            Label_[did]["out_sat_dist"], Label_[did]["out_brt_dist"])
    if save is not None:
        with open(os.path.abspath(save), 'wb') as f:
            pickle.dump(Label_, f)
            f.close()

    return Label_


# Add own features - Add own feature values
# Input: Feature values:Dict(Dict()) - append the own feature values into each defect. -> for each defect_id and for each own feature
# Output: Feature values:Dict(Dict()) - List of feature values after adding own features
def addOwnFeatures(feature, save=None):
    global Label_, feature_list

    feature_list.extend(list(feature[list(feature.keys())[0]].keys()))
    for did in feature.keys():
        for f in feature[did].keys():
            Label_[did][f] = feature[did][f]

    if save is not None:
        with open(os.path.abspath(save), 'wb') as f:
            pickle.dump(Label_, f)
            f.close()

    return Label_


# ====================Get Function Areas====================

# Get Image Directory
# Output:   Path String
def get_ImageDir():
    return Image_Dir


# Get Image Lists
# Output:   List
def get_ImageList():
    return Image_List


# Get Extracted Label
# Output:   Dictionary
def get_Label():
    return Label


# Get Extracted Features
# Output:   Dictionary
def get_Features():
    return Label_


def get_FeatureList():
    return feature_list


def get_FeatureRange(own_range=None):
    feature_range = {
        "size": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "coverage": [1, 2, 3, 4, 5],
        "asp_ratio": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "deg_avg": [1, 2, 3, 4, 5, 6],
        "deg_mode": [1, 2, 3, 4, 5, 6],
        "edge": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "edgelen_avg": [1, 2, 3, 4, 5],
        "edgelen_mode": [1, 2, 3, 4, 5],
        "distance": [1, 2, 3],
        "sc_edge_ratio": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "sc_follow_turn": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "sc_reverse_turn": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "sc_small_turn": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "sc_score": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "hue_avg": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "hue_mode": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "hue_range": [1, 2, 3, 4, 5, 6],
        "hue_uni": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "hue_outin": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "sat_avg": [1, 2, 3, 4, 5],
        "sat_mode": [1, 2, 3, 4, 5],
        "sat_range": [1, 2, 3, 4],
        "sat_uni": [1, 2, 3, 4, 5],
        "sat_outin": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "brt_avg": [1, 2, 3, 4, 5],
        "brt_mode": [1, 2, 3, 4, 5],
        "brt_range": [1, 2, 3, 4],
        "brt_uni": [1, 2, 3, 4, 5],
        "brt_outin": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "out_hue_avg": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "out_hue_mode": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "out_hue_range": [1, 2, 3, 4, 5, 6],
        "out_hue_uni": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "out_sat_avg": [1, 2, 3, 4, 5],
        "out_sat_mode": [1, 2, 3, 4, 5],
        "out_sat_range": [1, 2, 3, 4],
        "out_sat_uni": [1, 2, 3, 4, 5],
        "out_brt_avg": [1, 2, 3, 4, 5],
        "out_brt_mode": [1, 2, 3, 4, 5],
        "out_brt_range": [1, 2, 3, 4, 5],
        "out_brt_uni": [1, 2, 3, 4, 5],
    }
    if own_range is not None:
        for i in own_range.keys():
            feature_range[i] = own_range[i]

    return feature_range