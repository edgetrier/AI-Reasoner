import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import AIReasoner.TargetImporter as TI
import AIReasoner.FeatureExtraction as FE
import AIReasoner.Model as M
import AIReasoner.Plot as AIplot
import AIReasoner.AIOutputExporter as AIOE
import numpy as np
import cv2
import pickle

# [REQUIRE CHANGES] The directory path of your original images
original_images_dir = "./dataset/ori_images/"

# [REQUIRE CHANGES] The directory path of your ground truth masks
ground_truth_mask_dir = "./dataset/gt_masks"

# [REQUIRE CHANGES] The directory path of your prediction masks
prediction_dir = "./dataset/pred_masks"

# [REQUIRE CHECK] Does the masks also include the type classifications? If yes, please check your mask format
# Greyscaled Masks
# Shape: (H, W, 3) -> all channels' values should be same in every pixel or (H, W)
# The value starts 0 to the number of types (0 is background)


# Export prediction result follow the instruction which printed in the terminal

# Detection & Classification
mask_contain_type = True
only_type = False
prediction_data = AIOE.process_bydir(original_images_dir, ground_truth_mask_dir, prediction_dir, mask_contain_type, only_type)

# Detection - uncomment below if you only want to reason the detection task
# mask_contain_type = False
# only_type = False
#prediction_data = AIOE.process_bydir(original_images_dir, ground_truth_mask_dir, prediction_dir, mask_contain_type, only_type)

# Classification - uncomment below if you only want to reason the detection task
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
# for target in result.keys():
for target in ["wrong-type-classified"]:

    target_test = result[target]
    M.load_label_data(target_test, target)

    model, evaluation, error = M.plant_trees(feature, n_tree=200, reverse=False)

    good_tree, scores = M.val_trees(evaluation)

    path, node, route = M.climb_trees(model, feature_name=feature)

    tree_result = M.analyse_trees(path, node, error, feature, round_num=4)

    report, route_1, route_0, node = M.summary_trees(tree_result, feature, route, node, FE.get_FeatureRange())

    AIplot.model_plot(report, feature, route_1, "./AI-Reasoner Outputs/"+target, range_split=True, detail=True)

