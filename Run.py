import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import ForestMonkey.FeatureExtraction as FE
import ForestMonkey.Model as M
import ForestMonkey.Plot as AIplot
import ForestMonkey.AIOutputExporter as AIOE

import time


# [REQUIRE CHANGES] The directory path of your original images
original_images_dir = "./dataset/ori_imgs/"

# [REQUIRE CHANGES] The directory path of your ground truth masks
ground_truth_mask_dir = "./dataset/gt_masks/"

# [REQUIRE CHANGES] The directory path of your prediction masks
prediction_dir = "./dataset/pred_masks/"


# [REQUIRE CHECK] Does the masks also include the type classifications? If yes, please check your mask format
# Greyscaled Masks
# Shape: (H, W, 3) -> all channels' values should be same in every pixel or (H, W)
# The value starts 0 to the number of types (0 is background)


# Export prediction result follow the instruction which printed in the terminal

# Detection & Classification
mask_contain_type = True
only_type = False
prediction_data = AIOE.process_bydir(original_images_dir, ground_truth_mask_dir, prediction_dir, mask_contain_type, only_type)

# Detection ONLY - uncomment below if you only want to reason the detection task
# mask_contain_type = False
# only_type = False
#prediction_data = AIOE.process_bydir(original_images_dir, ground_truth_mask_dir, prediction_dir, mask_contain_type, only_type)

# Classification ONLY - uncomment below if you only want to reason the detection task
# mask_contain_type = True
# only_type = True
#prediction_data = AIOE.process_bydir(original_images_dir, ground_truth_mask_dir, prediction_dir, mask_contain_type, only_type)

# Generate the ground truth
# ground_truth_mask_data = {}
# for i in list(os.listdir(ground_truth_mask_dir)):
#     ground_truth_mask_data[i] = {"polygons":[]}
#     ground_truth_mask_data[i]["polygons"], _ = AIOE.process_mask(os.path.join(ground_truth_mask_dir, i))

# Generate Defect Characteristics
FE.checkImages(original_images_dir)
FE.readLabel(prediction_data)
FE.loadData(crop_pad=0.25, size_percent=True)
label = FE.featureExtract(outside="full", original=True, norm=True)
# merged_label = FE.mergeDefect_img(label)



id_list, feature, data, result = M.convert2List(label, FE.feature_list, prediction_data)

# Load Defect Characteristics in AI-Reasoner
M.load_feature_data(data)

# Start train AI-Reasoner
for target in result.keys():

    print(target)
    target_test = result[target]
    M.load_target_data(target_test, target)

    
    model = M.plant_forest(n_tree=200, reverse=False)

    good_tree, scores, error = M.val_forest(model, feature)

    path, node, route = M.climb_forest(model, feature_name=feature)

    analysed_nodes = M.analyse_forest(path, node, error, round_num=4)

    report, route_1, route_0 = M.summary_forest(analysed_nodes, route, feature, FE.get_FeatureRange())

    AIplot.explain_forest(report, feature, "./Performance/COVID/"+target, route_1, range_split=True, detail=True)
