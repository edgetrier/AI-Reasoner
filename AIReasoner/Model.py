import logging, os, random

import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import confusion_matrix as CM
import sklearn.tree as T
from sklearn.base import clone
import math
from tqdm import tqdm
import copy

Feature_Data = []
Label_Data = []
rev = False
val_rate = {"validation": 0, "TNR": 0, "TPR": 0}
target = ""

logging.getLogger().setLevel(logging.INFO)


# Transfer the feature values to defect_id list, feature name list and data list
# Input: label - feature values
# Output: Defect_id:List()
#         Feature name:List()
#         Data:List()
def convert2List(label, feature, output):
    data = []
    id_list = []
    result = {}
    for i in label.keys():
        id_list.append(i)
        e = []
        for j in feature:
            e.append(label[i][j])
        data.append(e)

        if label[i]["filename"] in output.keys():
            idx = label[i]["did"] - 1
            for r in output[label[i]["filename"]]["results"][idx]:
                if r not in result.keys():
                    result[r] = []
                result[r].append(output[label[i]["filename"]]["results"][idx][r])
        else:
            raise KeyError("No image file called:" + label[i]["filename"])


    return id_list, feature, data, result

def convert2List_old(label, feature, test):
    data = []
    id_list = []
    result = {"detection": [], "miss-detection": [], "type": [], "miss-type": [], "miss-type-strict": []}

    for i in label.keys():
        id_list.append(i)
        e = []
        for j in feature:
            e.append(label[i][j])
        data.append(e)
        if "detection" in test.keys():
            result["detection"].append(int(test["detection"][i]))
            if int(test["detection"][i]) == 1:
                result["miss-detection"].append(0)
            else:
                result["miss-detection"].append(1)
        if "type" in test.keys():
            result["type"].append(int(test["type"][i]))
            if int(test["type"][i]) == 1:
                result["miss-type"].append(0)
                result["miss-type-strict"].append(0)
            else:
                result["miss-type"].append(1)
                if int(test["detection"][i]) == 0:
                    result["miss-type-strict"].append(0)
                else:
                    result["miss-type-strict"].append(1)

        for j in test.keys():
            if j not in ["detection", "type"]:
                if j not in result.keys():
                    result[j] = []
                result[j].append(int(test[j][i]))

    return id_list, feature, data, result


# Load feature data into model
# Input: Feature data
def load_feature_data(data):
    global Feature_Data
    Feature_Data = data


# Load Test data into model
# Input: Test data
def load_label_data(data, targ):
    global Label_Data, target
    target = targ
    Label_Data = data


# Check all data is correctly loaded
def check_data():
    global Feature_Data, Label_Data
    if Feature_Data is None or len(Feature_Data) <= 0:
        logging.error("Feature_Data is missing or not correctly loaded")
    if Label_Data is None or len(Label_Data) <= 0:
        logging.error("Label_Data is missing or not correctly loaded")


# Plant trees - Training decision tree models
# Input: Feature name:List()
#        Number of decision tree:Integer - default:1
#        Criterion:"entropy" or "gini"
# Output: Trained model:sklearn
#         Evaluation matrix:[[tp,fp],[fn,tn]]
#         Error feature:[feature name]
def plant_trees(n_tree=1, criterion="gini", reverse=False, seeds=None):
    global Feature_Data, Label_Data, rev
    check_data()

    rev = reverse

    models = []
    # dt = DTC(criterion=criterion, splitter="random", max_features=1, max_depth=None, random_state=seeds)
    for i in tqdm(range(n_tree), desc="Plant trees"):
        # dt_c = clone(dt)
        dt_c = DTC(criterion=criterion, splitter="random", max_features=1, max_depth=None, random_state=random.randint(3333,6666))
        dt_m = dt_c.fit(Feature_Data, Label_Data)
        models.append(dt_m)
    return models


# Find all routes from the trees
# Input: Path:List() - Extracted path from trees
# Output: Routes:List()
def find_all_route(path):
    route = []
    temp = [[0, None]]
    for i in path:
        if i[0] == temp[-1][0]:
            temp[-1][-1] = True
            temp.append([i[1], None])

        else:
            route.append(copy.deepcopy(temp))
            temp = temp[:1 + temp.index([i[0], True])]
            temp[-1][-1] = False
            temp.append([i[1], None])

    route.append(copy.deepcopy(temp))

    return route


# Climb tree - Extract text-based tree to machine understandable arrays
# Input: Generated tree:graphviz
# Output: Path:List() - tree paths and node condition
#         Node:Dict() - tree nodes and relevant information
#         Route:List() - tree routes based on path
def climb_tree(tree):
    tree1 = tree.split("\n")[:-1]
    path = []
    node = {}
    for i in tree1:
        try:
            int(i[0])
        except:
            continue
        if "->" in i:
            p = i.split(";")[0].split("[")[0].split("->")
            path.append((int(p[0]), int(p[1])))
        else:
            node_i = int(i.split('"')[0].split(" ")[0])
            n = i.split('"')[1].split("\\n")
            info = {}
            if n[0].startswith("gini") or n[0].startswith("entropy"):
                info["leaf"] = True
                info["criterion"] = float(n[0].split("=")[-1])
                info["samples"] = int(n[1].split("=")[-1])
                value = [int(float(n[2].split("=")[-1].split(",")[0][2:]))]
                for j in range(1, len(n[2].split("=")[-1].split(",")) - 1):
                    value.append(int(n[2].split("=")[-1].split(",")[j][1:]))
                value.append(int(float(n[2].split("=")[-1].split(",")[-1][:-1])))
                info["value"] = value
                info["score"] = -1
            else:
                info["leaf"] = False

                info["feature"] = n[0].split(" ")[0]
                info["threshold"] = float(n[0].split(" ")[-1])

                info["criterion"] = float(n[1].split("=")[-1])
                info["samples"] = int(n[2].split("=")[-1])

                value = [int(n[3].split("=")[-1].split(",")[0][2:])]
                for j in range(1, len(n[3].split("=")[-1].split(",")) - 1):
                    value.append(int(n[3].split("=")[-1].split(",")[j][1:]))
                value.append(int(n[3].split("=")[-1].split(",")[-1][:-1]))
                info["value"] = value
                info["score"] = -1

            node[node_i] = info

    route = find_all_route(path)

    for rs in route:
        depth = 0
        for r in rs:
            if "depth" not in node[r[0]]:
                node[r[0]]["depth"] = depth
            depth += 1

    return path, node, route


# Climb tree (Forest) - Extract text-based tree to machine understandable arrays
# Input: Generated mode - trained sklearn models
#        Feature name - feature name list
# Output: Path:List() - tree paths and node condition
#         Node:Dict() - tree nodes and relevant information
#         Route:List() - tree routes based on path
def climb_trees(model, feature_name):
    routes = []
    nodes = []
    paths = []
    for i in tqdm(model, desc="Climb trees"):
        p, n, r = climb_tree(T.export_graphviz(i, feature_names=feature_name))
        routes.append(r)
        nodes.append(n)
        paths.append(p)
    return paths, nodes, routes


# Validate Trees - are the trained trees satisfied to make analysis
# Input: Evaluation matrix
#        Pass rate:Float[0-1] - Evaluation score lowest bound
#        Class Importance:[1,1]- class weight
# Output: Validation:Boolean - is tree satisfied to continue
#         Score list:List() - good score, T score and range, F score and range
def val_trees(model, feature_name, pass_rate=0.95):
    global rev, val_rate
    t0_prec = []
    t1_prec = []
    good = 0
    errors = []

    for m in tqdm(model, desc="Validate trees"):
        error = []
        predict = m.predict(Feature_Data)
        for j in range(len(predict)):
            if predict[j] != Label_Data[j]:
                node_id = m.apply([Feature_Data[j]])[0]
                if node_id in m.tree_.children_left:
                    node_id = np.where(m.tree_.children_left == node_id)
                elif node_id in m.tree_.children_right:
                    node_id = np.where(m.tree_.children_right == node_id)
                else:
                    continue
                wrong_feature = int(m.tree_.feature[node_id])
                if feature_name[wrong_feature] not in error:
                    error.append(feature_name[wrong_feature])
        errors.append(error)

        i = CM(Label_Data, predict)
        t0_pr = 0
        if i[0][0] == 0:
            t0_prec.append(0.0)
        else:
            t0_pr = i[0][0] / sum(i[0])
            t0_prec.append(t0_pr)
        t1_pr = 0
        if len(i) == 1:
            logging.error("No targets found!")
            t1_prec.append(t1_pr)
            continue
        if i[-1][-1] == 0:
            t1_prec.append(0.0)
        else:
            t1_pr = i[-1][-1] / sum(i[1])
            t1_prec.append(t1_pr)

        good += t0_pr * 0.5
        good += t1_pr * 0.5


    t0_avg = sum(t0_prec) / len(t0_prec)
    t1_avg = sum(t1_prec) / len(t1_prec)
    t0_range = max(t0_prec) - min(t0_prec)
    t1_range = max(t1_prec) - min(t1_prec)
    good_rate = good / len(model)

    if rev:
        logging.info("Reversed prediction enable, change result 0 to 1")
        logging.info("{0}% rules of result=1 (target) are learned.".format(round(t0_avg * 100, 2)))
        logging.info("{0}% rules of result=0 are learned.".format(round(t1_avg * 100, 2)))
        val_rate["validation"] = good_rate
        val_rate["TPR"] = t0_avg
        val_rate["TNR"] = t1_avg
    else:
        logging.info("{0}% rules of result=0 are learned.".format(round(t0_avg * 100, 2)))
        logging.info("{0}% rules of result=1 (target) are learned.".format(round(t1_avg * 100, 2)))
        val_rate["validation"] = good_rate
        val_rate["TPR"] = t1_avg
        val_rate["TNR"] = t0_avg

    if good_rate >= pass_rate:
        logging.info("Trees are validated and passed")
        return True, [good_rate, (t0_avg, t0_range), (t1_avg, t1_range)]
    else:
        logging.warning(
            "Validation rate is lower than the pass requirements. Required: {0};  Current: {1}".format(
                pass_rate, round(good_rate, 3)))
        return False, [good_rate, (t0_avg, t0_range), (t1_avg, t1_range)], errors


# Find the node's children node ids
# Input: node:id
#        node list:Dict()
#        Path:List()
# output: Children:[c1, c2] - if leaf, then [None, None]
def find_children(n, no, p):
    if no[n]["leaf"]:
        return None, None
    else:
        children = list(filter(lambda x: x[0] == n, p))
        return children[0][-1], children[-1][-1]


# Determine the distinguish-ability status
# Input: t0 - True child distinguish score
#        t1 - False child distinguish score
# Output: level_text:String - Confirmation, reduction, half reduction
def determine_level(t0, t1):
    if (t0 == 0.0 and t1 == 1.0) or (t0 == 1.0 and t1 == 0.0):
        return "confirmation"
    elif t0 == t1:
        return "reduction"
    elif t0 == 1.0 or t0 == 0.0 or t1 == 0.0 or t1 == 1.0:
        return "confirmation"
    elif t0 == 0.5 or t1 == 0.5:
        return "half reduction"
    else:
        return "reduction"


# Calculate decision and distinguish scores
# Input: node value - mis-classification node values
#        true value - mis-classification true child values
#        false value - mis-classification false child values
# Output: Decision score
#         Distinguish score
#         (direction to true, direction to f)
#         (direction to misclassified, direction to classified)
#         decision to mis-classification
#         decision status
def tree_decision(n, t, f):
    global rev
    t0 = t[0] / n[0]
    t1 = t[-1] / n[-1]
    f0 = f[0] / n[0]
    f1 = f[-1] / n[-1]

    decision = (abs(t0 - f0) + abs(t1 - f1)) / 2
    distinguish = abs(t1 - t0)
    direction_t = (t[-1] - t[0]) / (t[0] + t[-1])
    direction_f = (f[-1] - f[0]) / (f[0] + f[-1])

    direction_0 = t0 - f0
    direction_1 = t1 - f1

    decision_status = determine_level(t0, t1)

    if rev:
        direction_t = (t[0] - t[-1]) / (t[0] + t[-1])
        direction_f = (f[0] - f[-1]) / (f[0] + f[-1])

    decision_tf = direction_t >= direction_f

    return decision, distinguish, (direction_t, direction_f), (direction_0, direction_1), decision_tf, decision_status


# Determine the distinguish degree
# Input: Value - Distinguish score
#        Threshold - boundary for full, strong, middle, weak and empty
# Output: Degree of distinguish:String - full, strong, middle, weak and empty
def degree_status(value, threshold):
    if value <= 0.0:
        return "empty"
    elif value >= 1.0:
        return "full"
    elif value < threshold[0]:
        return "weak"
    elif value < threshold[1]:
        return "middle"
    else:
        return "strong"


# Analyse trees and calculate scores
# Input: Path
#        Node
#        Error features
#        Feature name list
#        Round number:Integer - if 0 then no round, else round to the closest decimal value
#        degree_threshold - distinguish degree boundary
#        own_result
# Output: Result:Dict(): for each tree for each feature and result
def analyse_trees(path, node, error, feature, round_num=0, degree_threshold=(0.25, 0.5), own_result=None):
    global rev

    tree_result = []
    if own_result is not None:
        if len(node) != len(own_result):
            raise ValueError("Own results do not match the generated trees structure.\n There is/are {0} trees, "
                             "but get {1} trees in own result".format(len(node), len(own_result)))

    for i in tqdm(range(len(node)), desc="Analyse trees"):
        total = sum(node[i][0]["value"])
        feature_result = {}
        for f in feature:
            feature_result[f] = []
        for j in node[i].keys():
            t, n = find_children(j, node[i], path[i])

            if t is None:
                continue

            value = node[i][j]["value"]
            true = node[i][t]["value"]
            false = node[i][n]["value"]
            decision, distinguish, direction_tf, direction_01, decision_tf, decision_status = \
                tree_decision(value, true, false)
            usage = sum(value) / total
            status_degree = degree_status(distinguish, degree_threshold)

            criterion = node[i][j]["criterion"]
            if round_num > 0:
                distinguish = round(distinguish, round_num)
                decision = round(decision, round_num)
                direction_tf = (round(direction_tf[0], round_num), round(direction_tf[1], round_num))
                direction_01 = (round(direction_01[0], round_num), round(direction_01[1], round_num))
                usage = round(usage, round_num)
            threshold = node[i][j]["threshold"]
            fea = node[i][j]["feature"]
            mistake = fea in error[i]
            result = {"tree": i,
                      "node": j,
                      "threshold": threshold,
                      "usage": usage,
                      "decision": decision,
                      "distinguish": distinguish,
                      "direction_tf": direction_tf,
                      "direction_01": direction_01,
                      "target_tf": decision_tf,
                      "status": decision_status,
                      "status_degree": status_degree,
                      "mistake": mistake}

            if own_result is not None:
                for result_name in own_result[i][j].keys():
                    if result_name in result:
                        raise KeyError("Result name duplicated: {0}".format(result_name))
                    else:
                        result[result_name] = own_result[i][j][result_name]

            feature_result[fea].append(result)
        tree_result.append(feature_result)

    return tree_result


# Calculate the rank of the feature based on the overall score
# Input: Feature score list:List()
# Output: Rank list:List() - the rank for each feature
def ranking(l):
    ranked = []
    rank = 0
    sort_l = list(sorted(copy.deepcopy(l), reverse=True))
    temp = {}
    for i in sort_l:
        if i not in temp:
            rank += 1
        temp[i] = rank

    for i in l:
        ranked.append(temp[i])

    return ranked


# Analyse threshold boundaries of each feature to locate mis-classification prediction based on the highest scores in one tree
# Input: Result list
#        Feature range - the min and max value of this feature
# Output: Lower bound and Upper bound
def threshold_split_1(result, feature_range):
    global rev
    lower = []
    upper = []
    max_b = feature_range["max"]
    min_b = feature_range["min"]
    re_t = sorted(list(filter(lambda x: x["target_tf"], result)), key=lambda x: x["score"], reverse=True)
    re_f = sorted(list(filter(lambda x: not x["target_tf"], result)), key=lambda x: x["score"], reverse=True)

    for i in re_t:
        if rev:
            if i["direction_01"][0] * i["distinguish"] * i["usage"] < 0:
                upper.append((i["threshold"], abs(i["direction_01"][0] * i["distinguish"] * i["usage"]) / 2))
            else:
                upper.append((i["threshold"], i["direction_01"][0] * i["distinguish"] * i["usage"]))
        else:
            if i["direction_01"][-1] * i["distinguish"] * i["usage"] < 0:
                upper.append((i["threshold"], abs(i["direction_01"][-1] * i["distinguish"] * i["usage"]) / 2))
            else:
                upper.append((i["threshold"], i["direction_01"][-1] * i["distinguish"] * i["usage"]))

    for i in re_f:
        if rev:
            if (0 - i["direction_01"][0]) * i["distinguish"] * i["usage"] < 0:
                lower.append((i["threshold"], abs((0 - i["direction_01"][0]) * i["distinguish"] * i["usage"]) / 2))
            else:
                lower.append((i["threshold"], (0 - i["direction_01"][0]) * i["distinguish"] * i["usage"]))
        else:
            if (0 - i["direction_01"][-1]) * i["distinguish"] * i["usage"] < 0:
                lower.append((i["threshold"], abs((0 - i["direction_01"][-1]) * i["distinguish"] * i["usage"]) / 2))
            else:
                lower.append((i["threshold"], (0 - i["direction_01"][-1]) * i["distinguish"] * i["usage"]))

    lower = sorted(lower, key=lambda x: x[-1], reverse=True)
    upper = sorted(upper, key=lambda x: x[-1], reverse=True)
    if len(lower) == 0 and len(upper) == 0:
        return (min_b, 0), (max_b, 0)
    elif len(lower) == 0:
        return (min_b, upper[0][1]), upper[0]
    elif len(upper) == 0:
        return lower[0], (max_b, lower[0][1])
    else:
        return lower[0], upper[0]


# Analyse the range/boundaries of the mis-classification of each feature in all trees
# Input: Lower bound and Upper bound
#        top_n:Integer - only calculate the top n scores
#        Max move: max threshold moving bound
# Output: Lower bound and Upper bound
#         Range score:Float - range moving gap between the initial and final range
#         Range stability:[Flat, Float] - the stability of range is changed every iteration
def bound_range_old(lower, upper, top_n, max_move=1):
    r = [lower[0][0], upper[0][0]]
    s = [lower[0][1], upper[0][1]]
    lower_stable = 0
    upper_stable = 0
    for i in range(top_n):
        lower_rate = (1 - (s[0] - lower[i][1]) / s[0]) / top_n
        if r[0] <= 0:
            if (lower[i][0] - r[0]) > 0:
                lower_gap = (lower[i][0] - r[0])
            else:
                lower_gap = 0
        else:
            lower_gap = (lower[i][0] - r[0]) / r[0]
        upper_rate = (1 - (s[1] - upper[i][1]) / s[1]) / top_n
        upper_gap = (upper[i][0] - r[1]) / r[1]
        lower_stable += (lower[i][0] - r[0]) / top_n
        upper_stable += (upper[i][0] - r[1]) / top_n
        if r[0] <= 0:
            if abs(lower_gap) > max_move:
                if lower_gap > 0:
                    lower_gap = max_move
                else:
                    lower_gap = 0
        else:
            if abs(lower_gap) > max_move / r[0]:
                if lower_gap > 0:
                    lower_gap = max_move / r[0]
                else:
                    lower_gap = 0 - max_move / r[0]

        if abs(upper_gap) > max_move / r[1]:
            if upper_gap > 0:
                upper_gap = max_move / r[1]
            else:
                upper_gap = 0 - max_move / r[1]

        r[0] += r[0] * lower_gap * lower_rate
        r[1] += r[1] * upper_gap * upper_rate

    range_score = 0
    if (r[0] + lower[0][0]) <= 0:
        range_score = abs(r[0] - lower[0][0])
    else:
        range_score = abs(r[0] - lower[0][0]) / (r[0] + lower[0][0])
    range_score += abs(r[1] - upper[0][0]) / (r[1] + upper[0][0])
    range_score /= 2
    return r, range_score, [lower_stable, upper_stable]


# Analyse the range/boundaries of the mis-classification of each feature in all trees
# Input: Lower bound and Upper bound
#        top_n:Integer - only calculate the top n scores
#        Max move: max threshold moving bound
# Output: Lower bound and Upper bound
#         Range score:Float - range moving gap between the initial and final range
#         Range stability:[Flat, Float] - the stability of range is changed every iteration
def bound_range(lower, upper, top_n, feature_r, max_move=0.05):
    r = [lower[0][0], upper[0][0]]
    s = [lower[0][1], upper[0][1]]
    lower_stable = 0
    upper_stable = 0
    for i in range(top_n):
        lower_rate = 0
        upper_rate = 0
        if s[0] > 0:
            lower_rate = (1 - (s[0] - lower[i][1]) / s[0]) * lower[i][1]
        if s[1] > 0:
            upper_rate = (1 - (s[1] - upper[i][1]) / s[1]) * upper[i][1]
        lower_move = lower_rate * (lower[i][0] - r[0])
        upper_move = upper_rate * (upper[i][0] - r[1])
        if abs(lower_move) > max_move * (feature_r["max"] - feature_r["min"]):
            if lower_move >= 0:
                lower_move = max_move
            else:
                lower_move = 0 - max_move
        if abs(upper_move) > max_move * (feature_r["max"] - feature_r["min"]):
            if upper_move >= 0:
                upper_move = max_move
            else:
                upper_move = 0 - max_move
        lower_stable += lower_move
        upper_stable += upper_move
        if (r[0] + lower_move) <= feature_r["min"]:
            r[0] = feature_r["min"]
        else:
            r[0] += lower_move

        if (r[1] + upper_move) >= feature_r["max"]:
            r[1] = feature_r["max"]
        else:
            r[1] += upper_move

    return r, [lower_stable, upper_stable]


# Calculate the overall score based on count and score - based on feature
# Input: Count and score list:List((count, score))
#        Count Score weight
# Output: Overall score:Float
def c_s_overall_feature(cs_list, c_s):
    overall = 0
    for i in cs_list:
        count = 1
        if (max([j[0] for j in cs_list]) - min([j[0] for j in cs_list])) > 0:
            count = (i[0] - min([j[0] for j in cs_list])) / (
                    max([j[0] for j in cs_list]) - min([j[0] for j in cs_list]))
        overall += count * c_s[0] + i[1] * c_s[1]
    overall /= len(cs_list)
    return overall


# Calculate the overall score based on count and score
# Input: Count and score list:List((count, score))
#        Count Score weight
# Output: Overall score:Float
def c_s_overall(count, score, c_s):
    return count * c_s[0] + score * c_s[1]


# Summary the analysed tree and calculate the final score and routes for all features and all trees
# Input: Analysed tree
#        Feature name
#        Routes
#        Nodes
#        Feature range
#        Feature weight:List() - default: "balance" or a list of integers and length must equal to feature name
#        Status bonus:List() - default:[0.2,0.5] - if the decision is confirmation or half reduction, score will get a bonus
#        Degree bonus:List() - default:[0,0.1,0.2,0.3,0.5] - if the decision is empty,weak,middle,strong,full, score will get a bonus
#        Mistake punish:Float() - default:2.0 - if decision make mistakes, the score will be reduced
#        All positive:Boolean - the score must be positive?
#        Own result
#        Top n:Integer - default:10 - pick top n score as range boundary
#        Count Score weight
# Output: Report:Dict() - report contain the score and relevant information need to write
#         Mis-classification routes
#         Classification routes
#         Node
def summary_trees(tree_result, feature, route, node, feature_range, feature_weight="balance", status_bonus=None,
                  degree_bonus=None, mistake_punish=2, all_positive=True, own_result=None, top_n=10, c_s=None):
    global rev
    if c_s is None:
        c_s = [0.25, 0.75]
    if status_bonus is None:
        status_bonus = [0.2, 0.5]
    else:
        if len(status_bonus) != 5:
            raise ValueError("Status bonus must have 2 values in the list")

    if degree_bonus is None:
        degree_bonus = [0, 0.1, 0.2, 0.3, 0.5]
    else:
        if len(degree_bonus) != 5:
            raise ValueError("Degree bonus must have 5 values in the list")
    weight = [1 for i in range(len(feature))]

    report = {}
    for f in feature:
        report[f] = {"feature": f,
                     "count": 0,
                     "score": 0,
                     "range": ([], []),
                     "overall": []}

    if feature_weight != "balance":
        if len(feature_weight) == len(feature):
            weight = feature_weight
        else:
            raise ValueError("feature weight list does not match the feature list")
    weight = [i / max(weight) for i in weight]
    for tree in tqdm(range(len(tree_result)), desc="Summarise trees - Brief Report"):
        fi = 0
        for f in feature:
            for fr in tree_result[tree][f]:

                score = fr["distinguish"]
                score *= 1 + fr["decision"]

                if fr["status"] == "half reduction":
                    score += status_bonus[0]
                elif fr["status"] == "confirmation":
                    score += status_bonus[1]

                if fr["status_degree"] == "empty":
                    score += degree_bonus[0]
                elif fr["status_degree"] == "weak":
                    score += degree_bonus[1]
                elif fr["status_degree"] == "middle":
                    score += degree_bonus[2]
                elif fr["status_degree"] == "strong":
                    score += degree_bonus[3]
                elif fr["status_degree"] == "full":
                    score += degree_bonus[4]

                if fr["mistake"]:
                    score -= mistake_punish

                if own_result is not None:
                    for i in own_result:
                        score += fr[i]

                if score < 0 and all_positive:
                    score = 0

                score *= fr["usage"]
                fr["score"] = score * weight[fi]
                node[fr["tree"]][fr["node"]]["score"] = score * weight[fi]

            scores = [i["score"] for i in tree_result[tree][f]]
            report[f]["count"] += len(scores) / len(tree_result)
            score_avg = 0
            if len(scores) == 0:
                report[f]["score"] += 0
            else:
                score_avg = sum(scores) / len(scores)
                report[f]["score"] += score_avg / len(tree_result)

            # report[f]["overall"].append([len(scores), score_avg])
            lower, upper = threshold_split_1(tree_result[tree][f], feature_range[f])
            report[f]["range"][0].append(lower)
            report[f]["range"][1].append(upper)
            report[f]["range"] = (sorted(report[f]["range"][0], key=lambda x: x[-1], reverse=True),
                                  sorted(report[f]["range"][1], key=lambda x: x[-1], reverse=True))

            fi += 1

    for f in report.keys():
        if top_n == 0:
            top_n = len(report[f]["range"][0])

        if top_n > len(report[f]["range"][0]):
            logging.warning("top_n value is larger than the number of trees. top_n: {0}, Trees: {1}\n\t"
                            "Replace the size of tree to the top_n".format(top_n, len(report[f]["range"][0])))
            top_n = len(report[f]["range"][0])
        report[f]["range"], report[f]["range_stable"] = bound_range(report[f]["range"][0], report[f]["range"][1], top_n,
                                                                    feature_range[f])
        # report[f]["overall"] = c_s_overall(report[f]["overall"], c_s)

    for f in report.keys():
        try:
            report[f]["count_norm"] = (report[f]["count"] - min([report[i]["count"] for i in report.keys()])) / (
                    max([report[i]["count"] for i in report.keys()]) - min(
                [report[i]["count"] for i in report.keys()]))
        except:
            report[f]["count_norm"] = 0.0
        try:
            report[f]["score_norm"] = (report[f]["score"] - min([report[i]["score"] for i in report.keys()])) / (
                    max([report[i]["score"] for i in report.keys()]) - min(
                [report[i]["score"] for i in report.keys()]))
        except:
            report[f]["score_norm"] = 0.0

    for f in report.keys():
        report[f]["overall"] = c_s_overall(report[f]["count_norm"], report[f]["score_norm"], c_s)

    for f in report.keys():
        try:
            report[f]["overall_norm"] = (report[f]["overall"] - min([report[i]["overall"] for i in report.keys()])) / (
                    max([report[i]["overall"] for i in report.keys()]) - min(
                [report[i]["overall"] for i in report.keys()]))
        except:
            report[f]["overall_norm"] = 0.0

    score_ranks = ranking([report[i]["overall"] for i in report.keys()])
    for i in range(len(score_ranks)):
        report[list(report.keys())[i]]["rank"] = score_ranks[i]

    route_1 = []
    route_0 = []
    for tree in tqdm(range(len(tree_result)), desc="Summarise trees - Route Report"):

        for r in route[tree]:
            score_r = [node[tree][i[0]]["score"] for i in r[:-1]]
            feature_r = [node[tree][i[0]]["feature"] for i in r[:-1]]
            threshold_r = [node[tree][i[0]]["threshold"] for i in r[:-1]]
            tf_r = [i[1] for i in r[:-1]]
            if len(score_r) > 0:
                overall_r = sum(score_r) / len(score_r)
            else:
                overall_r = 0.0
            decision_r = node[tree][r[-1][0]]["value"][1] >= node[tree][r[-1][0]]["value"][0]

            if rev:
                decision_r = not decision_r

            value_r = node[tree][r[-1][0]]["value"]
            length_r = len(r) - 1

            result_dir = {"overall": overall_r,
                          "decision": decision_r,
                          "value": value_r,
                          "length": length_r,
                          "score": score_r,
                          "feature": feature_r,
                          "threshold": threshold_r,
                          "tf": tf_r,
                          }
            if decision_r:
                result_dir["decision"] = 1
                route_1.append(result_dir)
            else:
                result_dir["decision"] = 0
                route_0.append(result_dir)

    return report, route_1, route_0, node


# Convert range to text
# Input: Feature range
#        Range
# Output: Text:String
def text_range(fr, r):
    rl = copy.deepcopy(fr)
    text = "<"
    count = 0
    for i in r:
        if i[0] < min(fr):
            rl.insert(0, None)
        elif i[0] > max(fr):
            rl.insert(len(fr), None)
        else:
            rl.insert(rl.index(i[0]), None)
        if i[1] > max(fr):
            rl.insert(len(fr), None)
        elif i[1] < min(fr):
            rl.insert(1, None)
        else:
            rl.insert(rl.index(i[1]) + 1, None)
    rl_clean = [rl[0]]

    for i in range(1, len(rl)):
        if rl[i] is None and rl[i - 1] is None:
            rl_clean = rl_clean[:-1]
        else:
            rl_clean.append(rl[i])

    start = True
    for i in rl_clean:
        if i is None:
            if start:
                text += "["
                start = not start
            else:
                text += "]"
                start = not start
        else:
            text += " " + str(i) + " "

    text += ">"
    return text


# Write route
# Input: Feature list
#        Threshold list
#        Decision list
# Output: Text:String
def write_route(feature, threshold, tf):
    text = ""
    for i in range(len(feature)):
        text += feature[i] + " "
        if tf[i]:
            text += "<="
        else:
            text += ">="
        text += " " + str(threshold[i]) + " "
        if i < len(feature) - 1:
            text += "and "
    return text


def route_explain(feature, threshold, tf, f_desc, f_range):
    text = ""
    for i in range(len(feature)):
        text += feature[i] + " "
        if tf[i]:
            text += "<="
        else:
            text += ">="
        value = round((threshold[i] - f_range[feature[i]]["min"]) /
                      (f_range[feature[i]]["max"] - f_range[feature[i]]["min"]) * max(f_desc[feature[i]]))
        text += " " + str(threshold[i]) + " [" + f_desc[feature[i]][value] + "] "
        if i < len(feature) - 1:
            text += "and "
    return text

def target_analyse(data, report, feature, target_name=""):
    prob = 0
    for f in feature:
        score = report[f]["overall_norm"]
        ranges = report[f]["range"]
        ranges_stable = report[f]["range_stable"]
        if ranges[1] < ranges[0]:
            if data[f] <= ranges[0] or data[f] >= ranges[1]:
                prob += score
            else:
                prob += (1 - min([abs(data[f] - ranges[0]), abs(data[f] - ranges[1])]) / abs(
                    ranges[1] - ranges[0])) * score * 0.5

        else:
            if data[f] >= ranges[0] or data[f] <= ranges[1]:
                prob += score
            else:
                prob += (1 - min([abs(data[f] - ranges[0]), abs(data[f] - ranges[1])]) / abs(
                    ranges[1] - ranges[0])) * score * 0.5

    return prob / len(feature)

def feature_explain(left, right, f_name, f_desc, f_range):
    text = "That means the defect's {name} is ".format(name=f_name)
    max_value = max(f_desc)
    min_value = min(f_desc)
    l_value = round((left - f_range["min"]) / (f_range["max"] - f_range["min"]) * max_value)
    r_value = round((right - f_range["min"]) / (f_range["max"] - f_range["min"]) * max_value)
    l_desc = f_desc[l_value]

    if l_value == min_value and r_value == max_value:
        return "That possibly means ANY value in the defect's {name}.".format(name=f_name)
    if l_value == r_value:
        text += l_desc
    elif l_value > r_value:
        for i in range(min_value, r_value):
            if f_desc[i] not in text:
                text += f_desc[i] + " or "
        if f_desc[r_value] not in text:
            text += f_desc[r_value]
        text += " OR "
        for i in range(l_value, max_value):
            if f_desc[i] not in text:
                text += f_desc[i] + " or "
        if f_desc[max_value] not in text:
            text += f_desc[max_value]
    else:
        for i in range(l_value, r_value):
            if f_desc[i] not in text:
                text += f_desc[i] + " or "
        if f_desc[r_value] not in text:
            text += f_desc[r_value]
    text += "."
    return text




# Convert the report to text-based string
# Input: Report
#        Mis-classification routes
#        Classification routes
#        Nodes
#        Feature range
#        Top routes:Integer - default:5 - print the top n routes based on the determining values
#        Stable move:Float - default:0.75 - Drawing a flexible range with discount/bonus move based on the stability
#        Save - save the text into a txt file
# Output: Text-based report:String
def explain_tree(report, route_1, route_0, node, feature_range, route_top=5, stable_warn=0.05, save=None):
    global rev, val_rate, target

    target_desc = {"detection": "correct detection results",
                   "miss-detection": "missed detection results",
                   "type": "correct type classification results",
                   "miss-type": "incorrect type classification results",
                   "miss-type-strict": "incorrect type classification results exclude missed detection"}

    hue_desc = {0: "red",
                1: "red",
                2: "red-orange",
                3: "orange",
                4: "yellow-orange",
                5: "yellow",
                6: "yello-green",
                7: "green",
                8: "green-blue",
                9: "blue",
                10: "blue-violet",
                11: "violet",
                12: "violet-red"}

    sat_desc = {0: "lost (pure grey)",
                1: "lower",
                2: "low",
                3: "strong",
                4: "stronger",
                5: "pure"}

    brt_desc = {0: "darkest",
                1: "darker",
                2: "dark",
                3: "bright",
                4: "brighter",
                5: "brightest"}

    hue_uni_desc = {0: "small (1-2)",
                    1: "small (1-2)",
                    2: "small (1-2)",
                    3: "few (3-4)",
                    4: "few (3-4)",
                    5: "some (5-6)",
                    6: "some (5-6)",
                    7: "many (7-8)",
                    8: "many (7-8)",
                    9: "various (9+)",
                    10: "various (9+)",
                    11: "various (9+)",
                    12: "various (9+)"}

    colour_uni_desc = {0: "small (1)",
                       1: "small (1)",
                       2: "few (2)",
                       3: "some (3)",
                       4: "many (4)",
                       5: "various (5)",
                       }

    hue_range_desc = {0: "no range (1)",
                      1: "low (1-2)",
                      2: "low (1-2)",
                      3: "large (3-4)",
                      4: "large (3-4)",
                      5: "wide (5+)",
                      6: "wide (5+)"}

    colour_range_desc = {0: "no range (1)",
                         1: "low (1-2)",
                         2: "low (1-2)",
                         3: "large (3-4)",
                         4: "large (3-4)",
                         5: "wide (5+)"}

    cc_desc = {0: "no difference",
               1: "small difference",
               2: "some difference",
               3: "many difference",
               4: "many difference",
               5: "large difference",
               6: "large difference",
               7: "large difference",
               8: "large difference",
               9: "large difference",
               10: "all difference"}

    cov_desc = {0: "empty",
                1: "smaller",
                2: "small",
                3: "large",
                4: "larger",
                5: "full"}

    deg_desc = {0: "sharp angle",
                1: "small acute angle",
                2: "acute angle",
                3: "right angle",
                4: "obtuse angle",
                5: "large obtuse angle",
                6: "similar as line"}

    edge_desc = {0: "few",
                 1: "some",
                 2: "many",
                 3: "many",
                 4: "too much"}

    sc_er_desc = {0: "really small (really complex)",
                  1: "small (complex)",
                  2: "neutral",
                  3: "large (simple)",
                  4: "really large (more simple)"}

    sc_ft_desc = {0: "really small (really complex)",
                  1: "small (complex)",
                  2: "neutral",
                  3: "large (simple)",
                  4: "really large (more simple)"}

    sc_rt_desc = {0: "really small (really simple)",
                  1: "small (simple)",
                  2: "neutral",
                  3: "large (complex)",
                  4: "really large (more complex)"}

    sc_st_desc = {0: "really small (really simple)",
                  1: "small (simple)",
                  2: "neutral",
                  3: "large (complex)",
                  4: "really large (more complex)"}

    size_desc = {0: "really small",
                 1: "small",
                 2: "neutral",
                 3: "large",
                 4: "really large"}

    asp_desc = {0: "really small (really simple)",
                1: "small (simple)",
                2: "large (complex)",
                3: "really large (more complex)",
                4: "really large (more complex)"}

    dist_desc = {0: "short",
                 1: "long",
                 2: "no neighbour defect"}

    feature_desc = {'out_hue_avg': hue_desc,
                    'out_hue_mode': hue_desc,
                    'out_hue_range': hue_range_desc,
                    'out_hue_uni': hue_uni_desc,
                    'out_sat_avg': sat_desc,
                    'out_sat_mode': sat_desc,
                    'out_sat_range': colour_range_desc,
                    'out_sat_uni': colour_uni_desc,
                    'out_brt_avg': brt_desc,
                    'out_brt_mode': brt_desc,
                    'out_brt_range': colour_range_desc,
                    'out_brt_uni': colour_uni_desc,
                    'size': size_desc,
                    'asp_ratio': asp_desc,
                    'distance': dist_desc,
                    'coverage': cov_desc,
                    'deg_avg': deg_desc,
                    'deg_mode': deg_desc,
                    'edge': edge_desc,
                    'sc_edge_ratio': sc_er_desc,
                    'sc_follow_turn': sc_ft_desc,
                    'sc_reverse_turn': sc_rt_desc,
                    'sc_small_turn': sc_st_desc,
                    'hue_avg': hue_desc,
                    'hue_mode': hue_desc,
                    'hue_range': hue_range_desc,
                    'hue_uni': hue_uni_desc,
                    'sat_avg': sat_desc,
                    'sat_mode': sat_desc,
                    'sat_range': colour_range_desc,
                    'sat_uni': colour_uni_desc,
                    'brt_avg': brt_desc,
                    'brt_mode': brt_desc,
                    'brt_range': colour_range_desc,
                    'brt_uni': colour_uni_desc,
                    'hue_outin': cc_desc,
                    'sat_outin': cc_desc,
                    'brt_outin': cc_desc}

    feature_name = {'out_hue_avg': "Average Hue (outside of defect area)",
                    'out_hue_mode': "Mode of Hue (outside of defect area)",
                    'out_hue_range': "Hue Range (outside of defect area)",
                    'out_hue_uni': "Number of Unique Hue (outside of defect area)",
                    'out_sat_avg': "Average Saturation (outside of defect area)",
                    'out_sat_mode': "Mode of Saturation (outside of defect area)",
                    'out_sat_range': "Saturation Range (outside of defect area)",
                    'out_sat_uni': "Number of Unique Saturation (outside of defect area)",
                    'out_brt_avg': "Average Brightness (outside of defect area)",
                    'out_brt_mode': "Mode of Brightness (outside of defect area)",
                    'out_brt_range': "Brightness Range (outside of defect area)",
                    'out_brt_uni': "Number of Unique Brightness (outside of defect area)",
                    'size': "Defect Size",
                    'asp_ratio': "Aspect Ratio",
                    'distance': "Distance",
                    'coverage': "Coverage",
                    'deg_avg': "Average Turning Angles",
                    'deg_mode': "Mode of Turning Angles",
                    'edge': "Number of Edge",
                    'sc_edge_ratio': "Edge Ratio",
                    'sc_follow_turn': "Number of Follow Turn",
                    'sc_reverse_turn': "Number of Reverse Turn",
                    'sc_small_turn': "Number of Small Turn",
                    'hue_avg': "Average Hue",
                    'hue_mode': "Mode of Hue",
                    'hue_range': "Hue Range",
                    'hue_uni': "Number of Unique Hue",
                    'sat_avg': "Average Saturation",
                    'sat_mode': "Mode of Saturation",
                    'sat_range': "Saturation Range",
                    'sat_uni': "Number of Unique Saturation",
                    'brt_avg': "Average Brightness",
                    'brt_mode': "Mode of Brightness",
                    'brt_range': "Brightness Range",
                    'brt_uni': "Number of Unique Brightness",
                    'hue_outin': "Hue Difference between Defect's Outside and Inside",
                    'sat_outin': "Saturation Difference between Defect's Outside and Inside",
                    'brt_outin': "Brightness Difference between Defect's Outside and Inside"}

    output_text = "AI Reasoner Report\n\n"
    output_text += "===Overview===\n"
    output_text += "AI Reasoner explains the decision-making process and defect characteristics in a defect " \
                   "classifier/detector.\n"
    output_text += "AI Reasoner Target: *{td}*.\n" \
                   "Reversed Prediction: *{rp}*.\n".format(td=target_desc[target], rp=rev)
    if rev:
        output_text += "Note: AI Reasoner analyses the model through focusing the label=0.\n"
    output_text += "Validation: *{v}%* defects have been correctly reasoned; \n" \
                   "*{tpr}%* defects on target have been correctly reasoned;\n" \
                   "*{tnr}%* defects on non-target have been correctly reasoned.\n".format(
        v=round(val_rate["validation"] * 100),
        tpr=round(val_rate["TPR"] * 100),
        tnr=round(val_rate["TNR"] * 100))

    output_text += "-Top Effect Defect Characteristics-\n"
    for i in sorted(report.values(), key=lambda x: x["rank"]):
        if i["overall_norm"] < 0.9:
            continue
        f_name = i["feature"]
        left = i["range"][0]
        right = i["range"][1]
        fr = feature_range[i["feature"]]
        desc = feature_explain(left, right, feature_name[f_name], feature_desc[f_name], fr)
        output_text += "Feature: {feature} - Range: {rang} Description: {desc}\n".format(
            feature=feature_name[i["feature"]],
            rang=[left, right],
            desc=desc)
    output_text += "-Top Effect Rules-\n"
    route_brief = sorted(route_1, key=lambda x: x["overall"], reverse=True)
    count = 0
    route_list = []
    while len(route_list) < route_top:
        i = route_brief[count]

        if i["feature"] in route_list:
            count += 1
            continue

        output_text += "Route: " + route_explain(i["feature"], i["threshold"], i["tf"], feature_desc, feature_range) \
                       + "\n"
        route_list.append(i["feature"])
        count += 1

    output_text += "===Defect Characteristics Ranks===\n"
    for i in sorted(report.values(), key=lambda x: x["rank"]):
        f_name = i["feature"]
        left = i["range"][0]
        right = i["range"][1]
        fr = feature_range[i["feature"]]
        ls = i["range_stable"][0] / (fr["max"] - fr["min"])
        rs = i["range_stable"][1] / (fr["max"] - fr["min"])

        desc = "The target will achieved in the detector/classifier when this feature's value is "
        if left > right:
            desc += "larger than {l} or smaller than {r} (two areas).".format(l=left, r=right)
        else:
            desc += "larger than {l} and smaller than {r} (one area).".format(l=left, r=right)

        desc += feature_explain(left, right, feature_name[f_name], feature_desc[f_name], fr) + "\n"

        if abs(ls) >= stable_warn and abs(rs) >= stable_warn:
            desc += "NOTE: The both range boundary is not stable, "
            if ls < 0 and rs < 0:
                desc += "the upper and lower boundary might be further lower"
            elif ls >= 0 and rs >= 0:
                desc += "the upper and lower boundary might be further higher"
            elif ls >= 0 and rs < 0:
                desc += "the upper boundary might be further lower and lower boundary might be further higher"
            elif ls < 0 and rs >= 0:
                desc += "the upper boundary might be further higher and lower boundary might be further lower"

        elif abs(ls) >= stable_warn:
            desc += "NOTE: The lower range boundary is not stable, "
            if ls < 0:
                desc += "the lower boundary might be further lower"
            else:
                desc += "the lower boundary might be further higher"
        elif abs(rs) >= stable_warn:
            desc += "NOTE: The upper range boundary is not stable, "
            if rs < 0:
                desc += "the upper boundary might be further lower"
            else:
                desc += "the upper boundary might be further higher"

        output_text += "Rank: {rank} - Feature: {feature} - DOS: {overall_norm} ({overall}) " \
                       "- DUF: {count_norm} ({count}) - DIS: {score_norm} ({score}) " \
                       "- Range:{rang} - Range Stability: {range_s}\n" \
                       "Description: {desc}" \
                       "\n---------\n".format(rank=i["rank"],
                                              feature=feature_name[i["feature"]],
                                              overall=round(i["overall"] * 100, 2),
                                              overall_norm=round(i["overall_norm"] * 100, 2),
                                              count=round(i["count"], 1),
                                              count_norm=round(i["count_norm"] * 100, 2),
                                              score=round(i["score"] * 100, 2),
                                              score_norm=round(i["score_norm"] * 100, 2),
                                              rang=[left, right],
                                              range_s=[ls, rs],
                                              desc=desc)
    output_text += "\n===Decision Routes===\n"
    output_text += "\n--Top DOS--\n"
    route_1_overall = sorted(route_1, key=lambda x: x["overall"], reverse=True)
    count = 0
    route_list = []
    while len(route_list) < route_top:
        i = route_1_overall[count]

        if i["feature"] in route_list:
            count += 1
            continue

        output_text += "Rank: {rank} - DOS: {overall} - Value: {value} - " \
                       "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"],
                                                          value=i["value"][1], length=i["length"])
        output_text += route_explain(i["feature"], i["threshold"], i["tf"], feature_desc, feature_range) + "\n"
        output_text += "------------------\n"
        route_list.append(i["feature"])
        count += 1
    output_text += "--Top Shortest Length--\n"
    route_1_slen = sorted(route_1, key=lambda x: x["length"])
    count = 0
    route_list = []
    while len(route_list) < route_top:
        i = route_1_slen[count]

        if i["feature"] in route_list:
            count += 1
            continue

        output_text += "Rank: {rank} - DOS: {overall} - Value: {value} - " \
                       "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"],
                                                          value=i["value"][1], length=i["length"])
        output_text += route_explain(i["feature"], i["threshold"], i["tf"], feature_desc, feature_range) + "\n"
        output_text += "------------------\n"
        route_list.append(i["feature"])
        count += 1
    output_text += "--Top Longest Length--\n"
    route_1_llen = sorted(route_1, key=lambda x: x["length"], reverse=True)
    count = 0
    route_list = []
    while len(route_list) < route_top:
        i = route_1_llen[count]

        if i["feature"] in route_list:
            count += 1
            continue

        output_text += "Rank: {rank} - DOS: {overall} - Value: {value} - " \
                       "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"],
                                                          value=i["value"][1], length=i["length"])
        output_text += route_explain(i["feature"], i["threshold"], i["tf"], feature_desc, feature_range) + "\n"
        output_text += "------------------\n"
        route_list.append(i["feature"])
        count += 1
    output_text += "--Top Largest Values--\n"
    route_1_val = sorted(route_1, key=lambda x: x["value"][1], reverse=True)
    if rev:
        route_1_val = sorted(route_1, key=lambda x: x["value"][0], reverse=True)
    count = 0
    route_list = []
    while len(route_list) < route_top:
        i = route_1_val[count]

        if i["feature"] in route_list:
            count += 1
            continue
        if rev:
            output_text += "Rank: {rank} - DOS: {overall} - Value: {value} - " \
                           "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"],
                                                              value=i["value"][0], length=i["length"])
        else:
            output_text += "Rank: {rank} - DOS: {overall} - Value: {value} - " \
                           "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"],
                                                              value=i["value"][1], length=i["length"])
        output_text += route_explain(i["feature"], i["threshold"], i["tf"], feature_desc, feature_range) + "\n"
        output_text += "------------------\n"
        route_list.append(i["feature"])
        count += 1
    output_text += "\n===Definition Descriptions===\n"
    output_text += "1. Defect Characteristics Range Description (from the lowest value to the highest value)\n"
    for i in feature_desc:
        output_text += i + ": " + str(feature_desc[i].values()) + " | min value: " + str(feature_range[i]["min"]) + \
                       " | max value: " + str(feature_range[i]["max"]) + "\n"
    output_text += "------\n"
    output_text += "2. Score Description in the Defect Characteristics Ranks/Routes\n"
    output_text += "The summary section will show the importance of each feature.\n" \
                   "The rank is according to its count and score values.\n" \
                   "The features in higher ranks are relatively stronger, but not definitely, to explain " \
                   "the target reasons than others.\n" \
                   "Rank: the rank of the feature distinguish-ability\n---\n" \
                   "DOS: the overall score of the feature distinguish-ability.\n" \
                   "The higher value the better.\n---\n" \
                   "DUF: the number of the average occurrence which this feature is used in all trees\n" \
                   "The higher number presents that this feature is popular and partially important for " \
                   "distinguishing the target\n---\n" \
                   "DIS: the average score of the decision importance by this feature in all trees\n" \
                   "The higher value presents that this feature can distinguish the target better\n---\n" \
                   "Range: the approximate range of this feature might cause target\n---\n" \
                   "Range Stability: the average stability of the top_n ranges in each feature\n" \
                   "The lower value presents that the range is stable and the target is more accurate\n" \
                   "---\n" \
                   "Description: a short text to describe the target situation with range visualisation\n" \
                   "---\n" \
                   "Value: the number of defects satisfied the target on that decision paths\n---\n" \
                   "Length: the length of the decision paths \n---\n" \
                   "* Please refer the range and value to determine the reasons why the model make prediction to " \
                   "achieve the target\n"

    if save is not None:
        if save.endswith(".txt"):
            with open(os.path.abspath(save), "w") as file:
                file.write(output_text)
                file.close()
        else:
            with open(os.path.abspath(save) + "/output.txt", "w") as file:
                file.write(output_text)
                file.close()
    return output_text


# Convert the report to text-based string
# Input: Report
#        Mis-classification routes
#        Classification routes
#        Nodes
#        Feature range
#        Top routes:Integer - default:5 - print the top n routes based on the determining values
#        Stable move:Float - default:0.75 - Drawing a flexible range with discount/bonus move based on the stability
#        Save - save the text into a txt file
# Output: Text-based report:String
def explain_tree_old(report, route_1, route_0, node, feature_range, route_top=5, stable_move=0.75, save=None):
    global rev, val_rate

    output_text = "AI Reasoner Report\n\n"
    output_text += "===Overview===\n"

    output_text += "The summary section will show the importance of each feature.\n" \
                   "The rank is according to its count and score values.\n" \
                   "The features in higher ranks are relatively stronger, but not definitely, to explain " \
                   "the target reasons than others.\n" \
                   "Rank: the rank of the feature distinguish-ability\n---\n" \
                   "DOS: the overall score of the feature distinguish-ability.\n" \
                   "The higher value the better.\n---\n" \
                   "DUF: the number of the average occurrence which this feature is used in all trees\n" \
                   "The higher number presents that this feature is popular and partially important for " \
                   "distinguishing the target\n---\n" \
                   "DIS: the average score of the decision importance by this feature in all trees\n" \
                   "The higher value presents that this feature can distinguish the target better\n---\n" \
                   "Range: the approximate range of this feature might cause target\n---\n" \
                   "Range Stability: the average stability of the top_n ranges in each feature\n" \
                   "The lower value presents that the range is stable and the target is more accurate\n" \
                   "---\n" \
                   "Description: a short text to describe the target situation with range visualisation\n" \
                   "---\n" \
                   "* Please refer the range and value to determine the reasons why the model make prediction to " \
                   "achieve the target\n=============\n\n"
    for i in sorted(report.values(), key=lambda x: x["rank"]):
        left = i["range"][0]
        right = i["range"][1]
        ls = i["range_stable"][0]
        rs = i["range_stable"][1]
        fr = feature_range[i["feature"]]
        desc = "The result might be 1 happened when "
        if rev:
            desc = "The result might be 0 (true target) happened when "
        if round(left, 1) > round(right, 1):
            if (right + rs * stable_move) - (left + ls * stable_move) >= 0:
                if round(left + ls * stable_move) == round(right + rs * stable_move):
                    desc += "{f} value is around {v}, but, might contain mistakes. Range Visualisation: ".format(
                        f=i["feature"], v=round(left + ls * stable_move))
                    desc += text_range(fr, [[round(left + ls * stable_move), round(right + rs * stable_move)]])
                else:
                    if round(left + ls * stable_move) <= round(right + rs * stable_move):
                        desc += "{f} value is around between {v1} and {v2}, but, might contain mistakes. Range " \
                                "Visualisation: ".format(f=i["feature"], v1=round(left + ls * stable_move),
                                                         v2=round(right + rs * stable_move))
                        desc += text_range(fr, [[round(left + ls * stable_move), round(right + rs * stable_move)]])
                    else:
                        desc += "{f} value is around between {v1} and {v2}, but, might contain mistakes. Range " \
                                "Visualisation: ".format(f=i["feature"], v2=round(left + ls * stable_move),
                                                         v1=round(right + rs * stable_move))
                        desc += text_range(fr, [[round(right + rs * stable_move), round(left + ls * stable_move)]])
            else:
                if int(right + rs * stable_move) == int(left + ls * stable_move) + 1:
                    if (left + ls * stable_move) - (right + rs * stable_move) < 0.5:
                        desc += "{f} value is any possible value between all its range. Range Visualisation: ".format(
                            f=i["feature"])
                    else:
                        desc += "{f} value is any possible value between all its range, but might not include {v1}. " \
                                "Range Visualisation: ".format(f=i["feature"], v1=int(right + rs * stable_move))
                    desc += text_range(fr, [[min(fr), int(right + rs * stable_move)],
                                            [int(left + ls * stable_move) + 1, max(fr)]])
                else:
                    if int(right + rs * stable_move) == int(left + ls * stable_move):
                        desc += "{f} value is any possible value between all its range, but might not include {v1}. " \
                                "Range Visualisation: ".format(f=i["feature"], v1=int(right + rs * stable_move))
                    else:
                        desc += "{f} value <= {v1} or >= {v2}. Range Visualisation: ".format(
                            f=i["feature"], v1=int(right + rs * stable_move), v2=int(left + ls * stable_move) + 1)
                    desc += text_range(fr, [[min(fr), int(right + rs * stable_move)],
                                            [int(left + ls * stable_move) + 1, max(fr)]])
        else:
            if round(left + ls * stable_move) == round(right + rs * stable_move):
                desc += "{f} value is around {v}. Range Visualisation: ".format(f=i["feature"],
                                                                                v=round(left + ls * stable_move))
                desc += text_range(fr, [[round(left + ls * stable_move), round(right + rs * stable_move)]])
            else:
                if round(left + ls * stable_move) <= round(right + rs * stable_move):
                    desc += "{f} value is around between {v1} and {v2}. Range Visualisation: ".format(
                        f=i["feature"], v1=round(left + ls * stable_move), v2=round(right + rs * stable_move))
                    desc += text_range(fr, [[round(left + ls * stable_move), round(right + rs * stable_move)]])
                else:
                    desc += "{f} value is around between {v1} and {v2}, but might contain mistakes. Range " \
                            "Visualisation: ".format(f=i["feature"], v2=round(left + ls * stable_move),
                                                     v1=round(right + rs * stable_move))

                    desc += text_range(fr, [[round(right + rs * stable_move), round(left + ls * stable_move)]])

        if ls >= 0.5 and rs < 0.5:
            desc += "\nBut the lower boundary might be further lower/higher."
        elif ls < 0.5 and rs >= 0.5:
            desc += "\nBut the upper boundary might be further lower/higher."
        elif ls >= 0.5 and rs >= 0.5:
            desc += "\nBut the both boundaries might be further lower or higher."

        output_text += "Rank: {rank} - Feature: {feature} - DOS: {overall_norm} ({overall}) " \
                       "- DUF: {count_norm} ({count}) - DIS: {score_norm} ({score}) - Range: {rang} " \
                       "- Range Stability: {rang_stable} - Range Difference: {ranges} \nDescription: {desc}" \
                       "\n---------\n".format(
            rank=i["rank"],
            feature=i["feature"],
            overall=round(i["overall"] * 100, 2),
            overall_norm=round(i["overall_norm"] * 100, 2),
            count=round(i["count"], 1),
            count_norm=round(i["count_norm"] * 100, 2),
            score=round(i["score"] * 100, 2),
            score_norm=round(i["score_norm"] * 100, 2),
            rang=[round(i["range"][0], 1),
                  round(i["range"][1], 1)],
            ranges=round(i["range_score"], 2), desc=desc,
            rang_stable=[round(i["range_stable"][0], 1),
                         round(i["range_stable"][1], 1)])

    output_text += "\n===Route Report===\n"
    output_text += "Route report section will show the feature conditions which might cause mis-classification " \
                   "or correct classification\n" \
                   "The score of each route is calculated based on the score of each feature and " \
                   "the decision routes in all trees.\n" \
                   "This section will select top meaningful routes which are based on the route score, route length " \
                   "and route decision values.\n" \
                   "The report contains the route and its related overall score, length, final values and " \
                   "mis-classification decision\n-------------\n" \
                   "DOS: the average score of this route\n" \
                   "Decision: the decision of the mis-classificaiton on this route\n" \
                   "Value: the number of defects is classified by following this route\n" \
                   "Length: the route length\n=============\n"
    route_1_overall = sorted(route_1, key=lambda x: x["overall"], reverse=True)
    route_0_overall = sorted(route_0, key=lambda x: x["overall"], reverse=True)
    output_text += "\n===Route Score Rank===\n"
    count = 1
    output_text += "\n-----Result to 1-----\n"
    for i in route_1_overall[:route_top]:
        output_text += "Rank: {rank} - DOS: {overall} - Result?: {decision} - Value: {value} - " \
                       "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"], decision=i["decision"],
                                                          value=i["value"][0], length=i["length"])
        output_text += write_route(i["feature"], i["threshold"], i["tf"]) + "\n"
        output_text += "------------------\n"
        count += 1
    count = 1
    if rev:
        output_text += "\n-----Result to 0 (true target)-----\n"
    else:
        output_text += "\n-----Result to 0-----\n"
    for i in route_0_overall[:route_top]:
        output_text += "Rank: {rank} - DOS: {overall} - Result?: {decision} - Value: {value} - " \
                       "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"], decision=i["decision"],
                                                          value=i["value"][1], length=i["length"])
        output_text += write_route(i["feature"], i["threshold"], i["tf"]) + "\n"
        output_text += "------------------\n"
        count += 1
    output_text += "\n===Route Length===\n"
    count = 1
    output_text += "\n-----Result to 1-----\n"
    route_1_len = sorted(route_1, key=lambda x: x["length"], reverse=False)
    route_0_len = sorted(route_0, key=lambda x: x["length"], reverse=False)
    for i in route_1_len[:route_top]:
        output_text += "Rank: {rank} - DOS: {overall} - Result?: {decision} - Value: {value} - " \
                       "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"], decision=i["decision"],
                                                          value=i["value"][0], length=i["length"])
        output_text += write_route(i["feature"], i["threshold"], i["tf"]) + "\n"
        output_text += "------------------\n"
        count += 1
    count = 1
    if rev:
        output_text += "\n-----Result to 0 (true target)-----\n"
    else:
        output_text += "\n-----Result to 0-----\n"
    for i in route_0_len[:route_top]:
        output_text += "Rank: {rank} - DOS: {overall} - Result?: {decision} - Value: {value} - " \
                       "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"], decision=i["decision"],
                                                          value=i["value"][1], length=i["length"])
        output_text += write_route(i["feature"], i["threshold"], i["tf"]) + "\n"
        output_text += "------------------\n"
        count += 1
    output_text += "\n===Route Value===\n"
    route_1_value = sorted(route_1, key=lambda x: x["value"][1], reverse=True)
    route_0_value = sorted(route_0, key=lambda x: x["value"][0], reverse=True)
    count = 1
    output_text += "\n-----Result to 1-----\n"
    for i in route_1_value[:route_top]:
        output_text += "Rank: {rank} - DOS: {overall} - Result?: {decision} - Value: {value} - " \
                       "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"], decision=i["decision"],
                                                          value=i["value"][0], length=i["length"])
        output_text += write_route(i["feature"], i["threshold"], i["tf"]) + "\n"
        output_text += "------------------\n"
        count += 1
    count = 1
    if rev:
        output_text += "\n-----Result to 0 (true target)-----\n"
    else:
        output_text += "\n-----Result to 0-----\n"
    for i in route_0_value[:route_top]:
        output_text += "Rank: {rank} - DOS: {overall} - Result?: {decision} - Value: {value} - " \
                       "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"], decision=i["decision"],
                                                          value=i["value"][1], length=i["length"])
        output_text += write_route(i["feature"], i["threshold"], i["tf"]) + "\n"
        output_text += "------------------\n"
        count += 1

    output_text += "\n===Improvement Explanation===\n"
    output_text += "This section will explain the possible processes for improving the detection " \
                   "performance. However, the possible improvement solutions are flexible and not limited.\n" \
                   "Generally, increasing dataset will improve the detection performance\n----------\n"
    output_text += "Size:\n1. Increase dataset amount of the corresponding sized defect\n2. Enlarge the defect area\n"
    output_text += "Coverage:\n1. Improve detection model architecture\n2. Suitable image augmentations\n"
    output_text += "Aspect Ratio:\n1. Improve detection model architecture\n2. Suitable image augmentations\n" \
                   "3. Normalise aspect ratio\n"
    output_text += "Average Vertex Degree:\n1. Improve detection model architecture\n2. Increase image features\n"
    output_text += "Mode Vertex Degree:\n1. Improve detection model architecture\n2. Increase image features\n"
    output_text += "Number of Edge:\n1. Suitable image augmentations\n2. Improve detection model architecture\n"
    output_text += "Average Edge Length:\n1. Suitable image augmentations\n2. Improve detection model architecture\n"
    output_text += "Mode Edge Length:\n1. Suitable image augmentations\n2. Improve detection model architecture\n"
    output_text += "Neighbour Distance:\n1. Separate neighbour defect to individual image if mis-classification " \
                   "happen with closed neighbour\n2. Increase dataset of similar situation\n"
    output_text += "Shape Complexity (Edge Ratio, Follow Turn, Small Turn, Reverse Turn):\n" \
                   "1. Suitable image augmentations\n" \
                   "2. Improve detection model architecture\n" \
                   "3. Increase image features\n" \
                   "4. Increase dataset\n"
    output_text += "Average HUE:\n1. Increase dataset\n"
    output_text += "Mode HUE:\n1. Increase dataset\n"
    output_text += "HUE Range:\n1. Normalise colour\n2. Improve detection model architecture\n"
    output_text += "Number of Unique HUE:\n1. Normalise colour\n2. Improve detection model architecture\n3. Grey-scale image " \
                   "if value is large\n"
    output_text += "Average Saturation:\n1. Increase dataset\n2. Grey-scale image if value is large\n"
    output_text += "Mode Saturation:\n1. Increase dataset\n2. Grey-scale image if value is large\n"
    output_text += "Saturation Range:\n1. Normalise colour\n2. Grey-scale image if value is large\n"
    output_text += "Number of Unique Saturation:\nn1. Normalise colour\n2. Improve detection model architecture\n3. Grey-scale " \
                   "image if value is large\n"
    output_text += "Average Brightness:\n1. Increase dataset\n2. Suitable image pre-processing\n"
    output_text += "Mode Brightness:\n1. Increase dataset\n2. Suitable image pre-processing\n"
    output_text += "Brightness Range:\n1. Normalise colour\n"
    output_text += "Number of Unique Brightness:\n1. Normalise colour\n"
    output_text += "Colour Complexity (HUE, Saturation, and Brightness):\n1. Normalise histogram of the image\n" \
                   "2. Improve detection model architecture\n"
    output_text += "Mode Hue (Outside):\n1. Normalise colour\n2. Increase dataset\n3. Improve detection model " \
                   "architecture\n "
    output_text += "Mode Saturation (Outside):\n1. Normalise colour\n2. Increase dataset\n3. Adjust contrast or " \
                   "histogram\n"
    output_text += "Mode Brightness (Outside):\n1. Normalise colour\n2. Increase dataset\n3. Adjust contrast or " \
                   "histogram\n"

    if save is not None:
        if save.endswith(".txt"):
            with open(os.path.abspath(save), "w") as file:
                file.write(output_text)
                file.close()
        else:
            with open(os.path.abspath(save) + "/output.txt", "w") as file:
                file.write(output_text)
                file.close()

    return output_text

