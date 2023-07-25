from matplotlib import pyplot as plt
from matplotlib import patches as pt
import matplotlib as mpl
import random
import math
import os, logging
import itertools as it
from scipy import stats
import pickle
from polygenerator import random_polygon, random_convex_polygon, random_star_shaped_polygon

logging.getLogger().setLevel(logging.INFO)

color_bar = [mpl.cm.Purples, mpl.cm.Blues, mpl.cm.Greens, mpl.cm.Oranges, mpl.cm.Reds, mpl.cm.YlOrRd]
color = ["red", "orange", "gold", "yellowgreen", "green", "springgreen", "cyan", "dodgerblue", "blue", "blueviolet",  "magenta", "deeppink"]
calculate_hue = None
calculate_sb = None

feature_name = {'out_hue_avg': "Average Hue (background)",
                    'out_hue_mode': "Mode of Hue (background)",
                    'out_hue_range': "Hue Range (background)",
                    'out_hue_uni': "Number of Unique Hue (background)",
                    'out_sat_avg': "Average Saturation (background)",
                    'out_sat_mode': "Mode of Saturation (background)",
                    'out_sat_range': "Saturation Range (background)",
                    'out_sat_uni': "Number of Unique Saturation (background)",
                    'out_brt_avg': "Average Brightness (background)",
                    'out_brt_mode': "Mode of Brightness (background)",
                    'out_brt_range': "Brightness Range (background)",
                    'out_brt_uni': "Number of Unique Brightness (background)",
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
                    'hue_outin': "Hue Difference",
                    'sat_outin': "Saturation Difference",
                    'brt_outin': "Brightness Difference"}


def data_hist_plot(data, target, x_feature, feature=None):
    color_choice1 = random.randint(0, len(color)-1)
    color_choice2 = color_choice1 + 6 if color_choice1 + 6 < len(color) else color_choice1 - 6

    if len(data) != len(target):
        raise ValueError("data and target have different length.")

    if feature is None:
        if type(x_feature) != type(int()):
            raise TypeError("x_feature must be Int type.")
    else:
        if type(x_feature) not in [type(int()), type(str())]:
            raise TypeError("x_feature must be Int or Str types")

    x_0 = []
    x_1 = []

    for i in range(len(data)):
        if type(x_feature) == type(int()):
            if target[i]:
                x_1.append(data[i][x_feature])
            else:
                x_0.append(data[i][x_feature])


        else:
            x_idx = feature.index(x_feature)
            if target[i]:
                x_1.append(data[i][x_idx])
            else:
                x_0.append(data[i][x_idx])


    fig = plt.Figure(figsize=(2000/300, 2000/300), dpi=300)
    ax = fig.add_axes([0,0,1,1])


    ax.hist(x_0, bins=20, color=color[color_choice1])
    ax.hist(x_1, bins=20, color=color[color_choice2])

    ax.set_xlabel("X")
    ax.set_ylabel("Frequency")
    if feature is not None:
        if type(x_feature) == type(int()):
            ax.set_xlabel(feature_name[feature[x_feature]])
        else:
            ax.set_xlabel(feature_name[x_feature])

    ax.set_xlim(-0.05,1.05)
    ax.legend(["Non-Target", "Target"])

    return fig




def data_2d_scatter_plot(data, target, x_feature, y_feature, feature=None):
    color_choice1 = random.randint(0, len(color)-1)
    color_choice2 = color_choice1 + 6 if color_choice1 + 6 < len(color) else color_choice1 - 6

    if len(data) != len(target):
        raise ValueError("data and target have different length.")

    if feature is None:
        if type(x_feature) != type(int()) or type(y_feature) != type(int()):
            raise TypeError("x_feature and y_feature must be Int type.")
    else:
        if type(x_feature) != type(y_feature):
            raise TypeError("x_feature and y_feature must be same type. (Int or Str)")
        if type(x_feature) not in [type(int()), type(str())]:
            raise TypeError("x_feature and y_feature must be Int or Str types")

    x_0 = []
    y_0 = []
    x_1 = []
    y_1 = []

    for i in range(len(data)):
        if type(x_feature) == type(int()):
            if target[i]:
                x_1.append(data[i][x_feature])
                y_1.append(data[i][y_feature])
            else:
                x_0.append(data[i][x_feature])
                y_0.append(data[i][y_feature])

        else:
            x_idx = feature.index(x_feature)
            y_idx = feature.index(y_feature)
            if target[i]:
                x_1.append(data[i][x_idx])
                y_1.append(data[i][y_idx])
            else:
                x_0.append(data[i][x_idx])
                y_0.append(data[i][y_idx])

    fig = plt.Figure(figsize=(2000/300, 2000/300), dpi=300)
    ax = fig.add_axes([0,0,1,1])


    ax.scatter(x_0, y_0, c=color[color_choice1])
    ax.scatter(x_1, y_1, c=color[color_choice2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if feature is not None:
        if type(x_feature) == type(int()):
            ax.set_xlabel(feature_name[feature[x_feature]])
            ax.set_ylabel(feature_name[feature[y_feature]])
        else:
            ax.set_xlabel(feature_name[x_feature])
            ax.set_ylabel(feature_name[y_feature])
    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)
    ax.legend(["Non-Target", "Target"])

    return fig


def single_route_plot(route, detail=False):
    count=0
    rang_s = [0,0]
    feature = []
    range_f = {}
    score = {}
    color_choice = random.choice(color_bar)

    for i in range(len(route["feature"])):
        if route["feature"][i] not in feature:
            score[route["feature"][i]] = route["score"][i]
            if route["tf"][i]:
                range_f[route["feature"][i]] = [0, route["threshold"][i]]
            else:
                range_f[route["feature"][i]] = [route["threshold"][i], 1.0]
            feature.append(route["feature"][i])
        else:
            c_rang = range_f[route["feature"][i]]
            if route["score"][i] > score[route["feature"][i]]:
                score[route["feature"][i]] = route["score"][i]
            if route["tf"][i]:
                if route["threshold"][i] < c_rang[1]:
                    c_rang[1] = route["threshold"][i]
            else:
                if route["threshold"][i] > c_rang[0]:
                    c_rang[0] = route["threshold"][i]

            range_f[route["feature"][i]] = c_rang

        count += 1
    score = [score[i] for i in feature]
    score_norm = []
    try:
        score_norm = [0.2 + 0.8*(i-min(score))/(max(score)-min(score)) for i in score]
    except:
        score_norm = [1.0 for i in score]

    block_num = len(feature)
    c = 5
    r = int((block_num+4) / 5)
    width = (c+2)*60
    height = (r+1)*30*2

    ci = 0
    ri = 0

    def node_name(name):
        output = ""
        l_name = name.split(" ")
        line_length = 0
        for i in l_name:
            if line_length + len(i) + 1 < 15:
                output += i + " "
                line_length += len(i) + 1
            else:
                output += "\n" + i + " "
                line_length = len(i) + 1
        return output
    fig = plt.Figure(figsize=(2000/300,round(2000/width*height)/300), dpi=300)

    if detail:
        fig = plt.Figure(figsize=(2000/300,round(300+2000/width*height+500*block_num)/300), dpi=300)
    ax = None
    if detail:
        ax = fig.add_axes([0,(500*block_num+300)/(300+2000/width*height+500*block_num),1,(2000/width*height)/(300+2000/width*height+500*block_num)])
    else:
        ax = fig.add_axes([0,0,1,1])


    ax.imshow([[[1.0,1.0,1.0] for _ in range(width)] for _ in range(height)])
    block_width = 60*0.6
    block_height = 30*0.6

    condition = ""

    for i in range(block_num):
        ci = i % 5
        ri = int(i / 5)


        if range_f[feature[i]][0] > range_f[feature[i]][1]:
            condition = "<= " + str(range_f[feature[i]][1]) + "\nor\n>= " + str(range_f[feature[i]][0])
        elif range_f[feature[i]][0] < range_f[feature[i]][1]:
            condition = ">= " + str(range_f[feature[i]][0]) + "\nand\n<=" + str(range_f[feature[i]][1])
        else:
            condition = "=" + str(range_f[feature[i]][0])

        if ri % 2:
            rect = pt.Rectangle(((5-ci) * 60 + 60*0.2, 30*0.2 + ri * 60), block_width, block_height, color="w", ec=color_choice(score_norm[i]))
            ax.add_patch(rect)
            ax.text((5-ci) * 60 + 60*0.5, 30*0.5 + ri * 60, node_name(feature_name[feature[i]]),  ha="center", va="center", fontsize=4)
            if ci == 4:
                rect = pt.Rectangle((60*0.2, 30*0.2 + ri * 60), block_width, block_height, color="orange")
                ax.add_patch(rect)
                ax.text(60*0.5, 30*0.5 + ri * 60, "to Non-Target",  ha="center", va="center", fontsize=5)
                ax.arrow((5-ci) * 60 + 60*0.18,  (ri+0.25) * 60, -60*0.35, 0, width=1, color="orange", length_includes_head=True)
                if i < block_num-1:
                    ax.arrow(60*1.5,  (ri+0.418) * 60, 0, 30*1.30, width=1, color="lime", length_includes_head=True)
                    ax.text(60*1.55,  (ri+0.75) * 60, condition, ha="left", va="center", fontsize=4)

            else:
                rect = pt.Rectangle(((5-ci) * 60 + 60*0.2, 30*0.2 + (0.5+ri) * 60), block_width, block_height, color="orange")
                ax.add_patch(rect)
                ax.text((5-ci) * 60 + 60*0.5, 30*0.5 + (0.5+ri) * 60, "to Non-Target",  ha="center", va="center", fontsize=5)
                ax.arrow((5.5-ci) * 60,  (ri+0.418) * 60, 0, 30*0.30, width=1, color="orange", length_includes_head=True)
                if i < block_num-1:
                    ax.arrow((5-ci) * 60 + 60*0.18,  (ri+0.25) * 60, -60*0.35, 0, width=1, color="lime", length_includes_head=True)
                    ax.text((5-ci) * 60,  ri * 60, condition, ha="center", va="top", fontsize=4)

        else:
            rect = pt.Rectangle((60*0.2 + (ci+1) * 60, 30*0.2 + ri * 60), block_width, block_height, color="w", ec=color_choice(score_norm[i]))
            ax.add_patch(rect)
            ax.text(60*0.5 + (ci+1) * 60, 30*0.5 + ri * 60, node_name(feature_name[feature[i]]),  ha="center", va="center", fontsize=4)
            if ci == 4:
                rect = pt.Rectangle((60*0.2 + 6 * 60, 30*0.2 + ri * 60), block_width, block_height, color="orange")
                ax.add_patch(rect)
                ax.text(60*0.5 + 6 * 60, 30*0.5 + ri * 60, "to Non-Target",  ha="center", va="center", fontsize=5)
                ax.arrow((ci+2) * 60 - 60*0.18,  (ri+0.25) * 60, 60*0.35, 0, width=1, color="orange", length_includes_head=True)
                if i < block_num-1:
                    ax.arrow(60*5.5,  (ri+0.418) * 60, 0, 30*1.30, width=1, color="lime", length_includes_head=True)
                    ax.text(60*5.55,  (ri+0.75) * 60, condition, ha="left", va="center", fontsize=4)

            else:
                rect = pt.Rectangle((60*0.2 + (ci+1) * 60, 30*0.2 + (0.5+ri) * 60), block_width, block_height, color="orange")
                ax.add_patch(rect)
                ax.text(60*0.5 + (ci+1) * 60, 30*0.5 + (0.5+ri) * 60, "to Non-Target",  ha="center", va="center", fontsize=5)
                ax.arrow((ci+1.5) * 60,  (ri+0.418) * 60, 0, 30*0.30, width=1, color="orange", length_includes_head=True)
                if i < block_num-1:
                    ax.arrow((ci+2) * 60 - 60*0.18,  (ri+0.25) * 60, 60*0.35, 0, width=1, color="lime", length_includes_head=True)
                    ax.text((ci+2) * 60,  ri * 60, condition, ha="center", va="top", fontsize=4)

    if ri % 2:
        if ci == 4:
            rect = pt.Rectangle(((5-ci) * 60 + 60*0.2, 30*0.2 + (0.5+ri) * 60), block_width, block_height, color="lime")
            ax.add_patch(rect)
            ax.text((5-ci) * 60 + 60*0.5, 30*0.5 + (0.5+ri) * 60, "to Target\n- Reasoned\n" + str(route["value"][-1]) + " Examples",  ha="center", va="center", fontsize=5)
            ax.arrow((5.5-ci) * 60,  (ri+0.418) * 60, 0, 30*0.30, width=1, color="lime", length_includes_head=True)
            ax.text((5.55-ci) * 60,  (ri+0.5) * 60, condition, ha="left", va="center", fontsize=4)

        else:
            rect = pt.Rectangle(((4-ci) * 60 + 60*0.2, 30*0.2 + ri * 60), block_width, block_height, color="lime")
            ax.add_patch(rect)
            ax.text((4-ci) * 60 + 60*0.5, 30*0.5 + ri * 60, "to Target\n- Reasoned\n" + str(route["value"][-1]) + " Examples",  ha="center", va="center", fontsize=5)
            ax.arrow((5-ci) * 60 + 60*0.18,  (ri+0.25) * 60, -60*0.35, 0, width=1, color="lime", length_includes_head=True)
            ax.text((5-ci) * 60,  ri * 60, condition, ha="center", va="top", fontsize=4)
    else:
        if ci == 4:
            rect = pt.Rectangle((60*0.2 + (ci+1) * 60, 30*0.2 + (0.5+ri) * 60), block_width, block_height, color="lime")
            ax.add_patch(rect)
            ax.text(60*0.5 + (ci+1) * 60, 30*0.5 + (0.5+ri) * 60, "to Target\n- Reasoned\n" + str(route["value"][-1]) + " Examples",  ha="center", va="center", fontsize=5)
            ax.arrow((ci+1.5) * 60,  (ri+0.418) * 60, 0, 30*0.30, width=1, color="lime", length_includes_head=True)
            ax.text((ci+1.55) * 60,  (ri+0.5) * 60, condition, ha="left", va="center", fontsize=4)
        else:
            rect = pt.Rectangle((60*0.2 + (ci+2) * 60, 30*0.2 + ri * 60), block_width, block_height, color="lime")
            ax.add_patch(rect)
            ax.text(60*0.5 + (ci+2) * 60, 30*0.5 + ri * 60, "to Target\n- Reasoned\n" + str(route["value"][-1]) + " Examples",  ha="center", va="center", fontsize=5)
            ax.arrow((ci+2) * 60 - 60*0.18,  (ri+0.25) * 60, 60*0.35, 0, width=1, color="lime", length_includes_head=True)
            ax.text((ci+2) * 60,  ri * 60, condition, ha="center", va="top", fontsize=4)


    # Setting

    ax.tick_params(axis='x',which='both',left=False,right=False, top=False, bottom=False, labelbottom=False, labeltop=False)
    ax.tick_params(axis='y',which='both',left=False,right=False, top=False, bottom=False, labelright=False, labelleft=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax2 = None
    if detail:
        ax2 = fig.add_axes([0,(500*block_num+300)/(300+2000/width*height+500*block_num),0.5,(60/height*0.2)*((2000/width*height)/(300+2000/width*height+500*block_num))])
    else:
        ax2 = fig.add_axes([0,0,0.5,60/height*0.2])
    ax2.imshow([[i/120 for i in range(20, 120)] for _ in range(2)], cmap=color_choice)
    ax2.tick_params(axis='x',which='both',left=False,right=False, top=True, bottom=False, labelbottom=False, labeltop=True)
    ax2.tick_params(axis='y',which='both',left=False,right=False, top=False, bottom=False, labelright=False, labelleft=False)
    ax2.set_xlabel("Importance in route", fontsize=5)
    ax2.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=100/10, offset=0))
    ax2.set_xticklabels([str(i)+"%" for i in range(0,101,10)], ha="center", fontsize=5)
    ax2.xaxis.set_label_position("top")


    ax3 = None
    if detail:
        ax3 = fig.add_axes([0.55,(500*block_num+300)/(300+2000/width*height+500*block_num),0.5,(0.9*60/height)*((2000/width*height)/(300+2000/width*height+500*block_num))])
    else:
        ax3 = fig.add_axes([0.55,0,0.5,0.9*60/height])
    ax3.imshow([[[1.0,1.0,1.0] for i in range(160)] for _ in range(50)])
    rect = pt.Rectangle((5, 5), 40, 15, color="w", ec=color_choice(0.5))
    ax3.add_patch(rect)
    ax3.text(25,12.5,"Defect\nCharacteristic", ha="center", va="center", fontsize=5)
    ax3.text(50,12.5,"Decision Node", ha="left", va="center", fontsize=5)
    ax3.text(5,21,"* Boundary color indicates\nthe importance degree of the node", ha="left", va="top", fontsize=3.5)

    rect = pt.Rectangle((5, 30), 30, 15, color="lime")
    ax3.add_patch(rect)
    ax3.text(20,37.5,"Leaf to Target\n- Reasoned\nN Examples", ha="center", va="center", fontsize=5)
    ax3.text(5,46,"* N is the number of examples found by this route", ha="left", va="top", fontsize=3.5)

    rect = pt.Rectangle((40, 30), 30, 15, color="orange")
    ax3.add_patch(rect)
    ax3.text(55,37.5,"Leaf\nto Non-Target", ha="center", va="center", fontsize=5)


    ax3.text(97.5, 10,"Condition", ha="center", va="center", fontsize=5)
    ax3.arrow(85, 15, 25, 0, width=1, color="lime", length_includes_head=True)
    ax3.text(115,15,"Decision condition\nto apporach\nreasoning target", ha="left", va="center", fontsize=5)

    ax3.arrow(85, 35, 25, 0, width=1, color="orange", length_includes_head=True)
    ax3.text(115,35,"Decision to\napporach\nnon-reasoning\ntarget", ha="left", va="center", fontsize=5)


    ax3.spines['right'].set_visible(True)
    ax3.spines['top'].set_visible(True)
    ax3.spines['left'].set_visible(True)
    ax3.spines['bottom'].set_visible(True)
    ax3.tick_params(axis='x',which='both',left=False,right=False, top=False, bottom=False, labelbottom=False, labeltop=False)
    ax3.tick_params(axis='y',which='both',left=False,right=False, top=False, bottom=False, labelright=False, labelleft=False)

    if detail:
        ratio = 500/(300+2000/width*height+500*block_num)
        count = 1
        for i in feature:
            rang = range_f[i]
            rang_s = [0,0]
            name = feature_name[i]
            ax_d = fig.add_axes([0,(500*(block_num-count))/(300+2000/width*height+500*block_num),1,ratio])
            plot_dict[i](ax_d, rang, rang_s, dc_name=name)
            count += 1


    return fig


def specific_route_plot(route, index, key="value", rev=True, detail=False):
    r = []
    if key == "value":
        r = sorted(route, key=lambda x: (x["value"][-1], x["overall"]), reverse=rev)
    elif key == "length":
        r = sorted(route, key=lambda x: (x["length"], x["overall"]), reverse=rev)
    elif key == "score":
        r = sorted(route, key=lambda x: (x["overall"], x["value"]), reverse=rev)

    fig = single_route_plot(r[index], detail=detail)


    return fig


def summary_range_plot(report, feature, detail=False):
    if detail:
        fig = plt.Figure(figsize=(2000/300,500*len(feature)/300), dpi=300)
        sort_report = sorted([report[i] for i in feature], key=lambda x: x["overall"], reverse=True)
        count = 1
        for i in sort_report:
            ax = fig.add_axes([0,1-count/len(feature),1,1/len(feature)])
            ax = plot_dict[i["feature"]](ax, i["range"], i["range_stable"], dc_name=feature_name[i["feature"]], overall=str(round(i["overall_norm"]*100))+"%", score=str(round(i["score_norm"]*100))+"%", count=str(round(i["count"],2)))
            count += 1

        return fig
    else:
        fig = plt.Figure(figsize=(3500/300,3000/300), dpi=300)
        sort_report = sorted([report[i] for i in feature], key=lambda x: x["overall"], reverse=True)
        color_choice = color_bar.copy()
        random.shuffle(color_choice)
        feature_l = [str(round(i["overall_norm"]*100)) + "%" for i in sort_report]

        ax1 = fig.add_axes([0,0,1,1])
        range_area = []
        count = 0
        for i in sort_report:
            rang = [round(159*r) for r in i["range"]]
            rang_s = [round(159*r) for r in i["range_stable"]]
            score = i["score_norm"]
            if count > 0:
                range_area += [[[1.0,1.0,1.0,1.0] for _ in range(220)] for _ in range(2)]
            target_area = []
            note = False
            if rang[0] > rang[1]:
                lefts = rang[1]
                lefte = rang[1]
                rights = rang[0]
                righte = rang[0]

                if rang_s[1] >= 0:
                    lefts += rang_s[1]
                else:
                    lefte += rang_s[1]
                if rang_s[0] >= 0:
                    righte += rang_s[0]
                else:
                    rights += rang_s[0]

                if lefts > rights:
                    note = True
                    lefts = rang[1]
                    lefte = rang[1]
                    rights = rang[0]
                    righte = rang[0]


                for l in range(160):
                    if l <= lefte or l >= righte:
                        target_area.append(color_choice[0](0.3+0.7*score))
                    elif l <= lefts and l > lefte:
                        perc = (lefts - l) / (lefts - lefte)
                        target_area.append(color_choice[0](0.3+0.7*score*perc))
                    elif l >= rights and l < righte:
                        perc = (l - rights) / (righte - rights)
                        target_area.append(color_choice[0](0.3+0.7*score*perc))
                    else:
                        target_area.append(color_choice[0](0.1))
            else:
                lefts = rang[0]
                lefte = rang[0]
                rights = rang[1]
                righte = rang[1]
                if rang_s[0] >= 0:
                    lefte += rang_s[0]
                else:
                    lefts += rang_s[0]
                if rang_s[1] >= 0:
                    rights += rang_s[1]
                else:
                    righte += rang_s[1]

                if lefte > righte:
                    note = True
                    lefts = rang[0]
                    lefte = rang[0]
                    rights = rang[1]
                    righte = rang[1]

                for l in range(160):
                    if l >= lefte and l <= righte:
                        target_area.append(color_choice[0](0.3+0.7*score))
                    elif l >= lefts and l < lefte:
                        perc = (l - lefts) / (lefte - lefts)
                        target_area.append(color_choice[0](0.3+0.7*score*perc))
                    elif l <= rights and l > righte:
                        perc = (rights - l) / (rights - righte)
                        target_area.append(color_choice[0](0.3+0.7*score*perc))
                    else:
                        target_area.append(color_choice[0](0.1))

            range_area += [[[1.0,1.0,1.0,1.0] for _ in range(60)] + target_area for _ in range(4)]
            if note:
                ax1.text(0, 2+6*count, feature_name[i["feature"]]+"*", ha="left", va="center", fontsize=8)
            else:
                ax1.text(0, 2+6*count, feature_name[i["feature"]], ha="left", va="center", fontsize=8)

            for t in range(10):
                ax1.text(60+round(160*t/10), 4.5+6*count, "0."+str(t), ha="center", va="center", fontsize=6)

            ax1.text(220, 4.5+6*count, "1.0", ha="center", va="center", fontsize=6)

            count += 1


        ax1.imshow(range_area)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.tick_params(axis='x',which='both',left=False,right=False, top=False, bottom=False, labelbottom=False, labeltop=False)
        ax1.tick_params(axis='y',which='both',left=False,right=False, top=False, bottom=False, labelright=False, labelleft=False)
        # ax1.set_title("Defect Characteristic Range Summary Chart", fontweight="bold")

        ax3 = fig.add_axes([14/15,0,1/15,1])
        ax3.imshow([[color_choice[0](i/100) for _ in range(2)] for i in range(30,100)])
        ax3.tick_params(axis='x',which='both',left=False,right=False, top=False, bottom=False, labelbottom=False)
        ax3.tick_params(axis='y',which='both',left=False,right=True, top=False, bottom=False, labelright=True, labelleft=False)
        ax3.set_ylabel("DIS")
        ax3.yaxis.set_major_locator(mpl.ticker.IndexLocator(base=70/20, offset=0))
        ax3.set_yticklabels([str(i)+"%" for i in range(0,101,5)])
        ax3.yaxis.set_label_position("right")

        ax4 = fig.add_axes([8.5/15,-1.2/15,5/15,1/15])
        ax4.tick_params(axis='x',which='both',left=False,right=False, top=False, bottom=False, labelbottom=False)
        ax4.tick_params(axis='y',which='both',left=False,right=False, top=False, bottom=False, labelleft=False)

        legend_area = [[[1.0,1.0,1.0,1.0] for _ in range(28)]]
        legend_area += [[[1.0,1.0,1.0,1.0] for _ in range(2)] +
                        [color_choice[0](0.1) for _ in range(8)] +
                        [color_choice[0](0.1+i*0.5/7) for i in range(8)] +
                        [color_choice[0](0.6) for _ in range(8)] +
                        [[1.0,1.0,1.0,1.0] for _ in range(2)] for _ in range(2)]
        legend_area += [[[1.0,1.0,1.0,1.0] for _ in range(28)] for _ in range(2)]

        ax4.imshow(legend_area)
        ax4.text(2+3.5, 1.5, "Range to Non-Target", ha="center", va="center", fontsize=6)
        ax4.text(2+11.5, 1.5, "Unstability\non Range Bound", ha="center", va="center", fontsize=6)
        ax4.text(2+19.5, 1.5, "Range to Target", ha="center", va="center", fontsize=6)
        ax4.text(1, 3.5, "* = The range is not stable and contain intersection", ha="left", va="center", fontsize=8)


        return fig




def summary_plot(report, feature, c_s = [0.25, 0.75]):
    fig = plt.Figure(figsize=(3500/300,3000/300), dpi=300)
    sort_report = sorted([report[i] for i in feature], key=lambda x: x["overall"])
    color_choice = color_bar.copy()
    random.shuffle(color_choice)
    feature_l = [str(round(i["overall_norm"]*100)) + "%" for i in sort_report]

    ax1 = fig.add_axes([0,0,13/15,1])
    bar1 = ax1.barh(y=[feature_name[i["feature"]] for i in sort_report], width=[round(100*i["score_norm"]*c_s[1]) for i in sort_report], color=[color_choice[0](i["score_norm"]) for i in sort_report], linewidth=10)
    bar2 = ax1.barh(y=[feature_name[i["feature"]] for i in sort_report], width=[round(100*i["count_norm"]*c_s[0]) for i in sort_report], color=[color_choice[-1](i["count_norm"]) for i in sort_report], linewidth=10, left=[round(100*i["score_norm"]*c_s[1]) for i in sort_report])
    ax1.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=100/20, offset=0))
    ax1.set_xticklabels([str(i)+"%" for i in range(0,101,5)])
    ax1.bar_label(bar2, feature_l, padding=0)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylabel("Defect Characteristic")
    ax1.set_xlabel("DOS")
    # ax1.set_title("Summary Chart", fontweight="bold")



    ax2 = fig.add_axes([13/15,0,1/15,1])
    ax2.imshow([[i/100 for _ in range(2)] for i in range(0, 100)], cmap=color_choice[0])
    ax2.tick_params(axis='x',which='both',left=False,right=False, top=False, bottom=False, labelbottom=False)
    ax2.tick_params(axis='y',which='both',left=False,right=True, top=False, bottom=False, labelright=True, labelleft=False)
    ax2.set_ylabel("DIS")
    ax2.yaxis.set_major_locator(mpl.ticker.IndexLocator(base=100/20, offset=0))
    ax2.set_yticklabels([str(i)+"%" for i in range(0,101,5)], ha="left")
    ax2.yaxis.set_label_position("left")

    ax3 = fig.add_axes([14/15,0,1/15,1])
    ax3.imshow([[i/100 for _ in range(2)] for i in range(0, 100)], cmap=color_choice[-1])
    ax3.tick_params(axis='x',which='both',left=False,right=False, top=False, bottom=False, labelbottom=False)
    ax3.tick_params(axis='y',which='both',left=True,right=False, top=False, bottom=False, labelright=False, labelleft=False)
    ax3.set_ylabel("DUF")
    ax3.yaxis.set_major_locator(mpl.ticker.IndexLocator(base=100/20, offset=0))
    ax3.set_yticklabels([str(i)+"%" for i in range(0,101,5)])
    ax3.yaxis.set_label_position("right")

    ax4 = fig.add_axes([6/15,0.5/15,7/15,7/16*3/15])
    ax4.tick_params(axis='x',which='both',left=False,right=False, top=False, bottom=False, labelbottom=False)
    ax4.tick_params(axis='y',which='both',left=False,right=False, top=False, bottom=False, labelleft=False)
    legend_area = [[[1,1,1,1] for _ in range(16)], [[1,1,1,1]]+[color_choice[0](0.5) if i<5 else color_choice[-1](0.5) for i in range(10)]+[[1,1,1,1] for _ in range(5)], [[1,1,1,1] for _ in range(16)]]
    ax4.imshow(legend_area)
    ax4.text(1+2, 1, "DIS", ha="center", va="center")
    ax4.text(6+2, 1, "DUF", ha="center", va="center")
    ax4.text(8.75+2, 1, "=", ha="center", va="center")
    ax4.text(11.25+2, 1, "Normalised \nDOS (%)", ha="center", va="center")


    return fig

def hue_avg_md_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):


    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = ['red', 'orange', 'yellow', 'yellow-green', 'green', 'spring-green', 'cyan', 'sky-blue', 'blue', 'violet-purple', 'magenta', 'rose', "red"]

    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=mpl.cm.hsv)


    # Draw Mark Line
    line_length = 360 / (len(x_labels)-1)
    gap = 0.025 * line_length
    for i in range(0, len(x_labels)):
        if i == 0:
            ax.plot([0, line_length/2-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length/2-gap), 359] , [22, 22], c="k")
        else:
            start = line_length/2 + (i-1) * line_length + gap
            end = line_length/2 + i * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")

    # Left Bound
    ax.plot([left,left], [-1,20], c="k", linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k", linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k", linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k", linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k", linestyle="-")
        ax.plot([359,354], [20,20], c="k", linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k", linestyle="-")
        ax.plot([0,5], [20,20], c="k", linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, color="k", width=0.5, length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, color="k", width=0.5, length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=13))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax


def brt_avg_md_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    x_labels = [0,1,2,3,4,5]

    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=mpl.cm.binary_r)


    # Draw Mark Line
    line_length = 360 / (len(x_labels)-1)
    gap = 0.025 * line_length
    for i in range(0, len(x_labels)):
        if i == 0:
            ax.plot([0, line_length/2-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length/2-gap), 359] , [22, 22], c="k")
        else:
            start = line_length/2 + (i-1) * line_length + gap
            end = line_length/2 + i * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")

    # Draw Example

    for i in range(len(x_labels)):
        if x_labels[i] == 0:
            ax.text(line_length/4,-10, "darkest",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 1:
            ax.text(line_length*i,-10, "darker", ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(line_length*i,-10, "dark", ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 3:
            ax.text(line_length*i,-10, "bright", ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 4:
            ax.text(line_length*i,-10, "brighter", ha="center", va="center", fontsize=5)
        else:
            ax.text(359-line_length/4-5,-10, "brightest", ha="center", va="center", fontsize=5)

    # Left Bound
    ax.plot([left,left], [-1,20], c="dodgerblue",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="dodgerblue",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="dodgerblue", linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="dodgerblue", linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="dodgerblue", linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="dodgerblue", linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="dodgerblue", linestyle="-")
        ax.plot([359,354], [-1,-1], c="dodgerblue", linestyle="-")
        ax.plot([359,354], [20,20], c="dodgerblue", linestyle="-")

        ax.plot([0,0], [-1,20], c="dodgerblue", linestyle="-")
        ax.plot([0,5], [-1,-1], c="dodgerblue", linestyle="-")
        ax.plot([0,5], [20,20], c="dodgerblue", linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="dodgerblue", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="dodgerblue", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")
    return ax


def sat_avg_md_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    x_labels = [0,1,2,3,4,5]

    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    rgb_choice = random.randint(0,2)

    c = []
    for _ in range(20):
        c_r = []
        for i in range(360):
            rgb = [0,0,0]
            rgb[rgb_choice] = i/359
            c_r.append(rgb)
        c.append(c_r)
    ax.imshow(c)


    # Draw Mark Line
    line_length = 360 / (len(x_labels)-1)
    gap = 0.025 * line_length
    for i in range(0, len(x_labels)):
        if i == 0:
            ax.plot([0, line_length/2-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length/2-gap), 359] , [22, 22], c="k")
        else:
            start = line_length/2 + (i-1) * line_length + gap
            end = line_length/2 + i * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")


    # Draw Example:

    for i in range(len(x_labels)):
        if x_labels[i] == 0:
            ax.text(line_length/4,-10, "lost\ngreyscaled",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 1:
            ax.text(line_length*i,-10, "lower",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(line_length*i,-10, "low",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 3:
            ax.text(line_length*i,-10, "strong",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 4:
            ax.text(line_length*i,-10, "stronger",  ha="center", va="center", fontsize=5)
        else:
            ax.text(359-line_length/4,-10, "pure\nfull coloured",  ha="center", va="center", fontsize=5)

    # Left Bound
    ax.plot([left,left], [-1,20], c="dodgerblue",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="dodgerblue",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="dodgerblue",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="dodgerblue",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="dodgerblue",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="dodgerblue",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="dodgerblue",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="dodgerblue",  linestyle="-")
        ax.plot([359,354], [20,20], c="dodgerblue",  linestyle="-")

        ax.plot([0,0], [-1,20], c="dodgerblue",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="dodgerblue",  linestyle="-")
        ax.plot([0,5], [20,20], c="dodgerblue",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=1, color="dodgerblue", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=1, color="dodgerblue", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")

    return ax

def hue_uni_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [1,2,3,4,5,6,7,8,9,10,11,12]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")

    # Draw Example
    color_copy = color.copy()
    random.shuffle(color_copy)
    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)
        if x_labels[i] == 1:
            ax.text(start+line_length/2,-10, "mono-colored",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 3:
            ax.text(start+line_length/2,-10, "few colours",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 6:
            ax.text(start+line_length/2,-10, "some colours",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 8:
            ax.text(start+line_length/2,-10, "many colours",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "various colours",  ha="center", va="center", fontsize=5)

        if x_labels[i] <= 4:
            for j in range(1, x_labels[i]+1):
                c = x_labels[i]+1
                m = int(c/2)
                if j % 2:
                    circle = pt.Circle((box[0]+(m-int(j/2))*width/c, box[1]+height/2), radius=1, color=color_copy[j-1])
                    ax.add_patch(circle)
                else:
                    circle = pt.Circle((box[0]+(m+int(j/2))*width/c, box[1]+height/2), radius=1, color=color_copy[j-1])
                    ax.add_patch(circle)
        elif x_labels[i] <= 8:
            for j in range(1, 5):
                c = 5
                m = int(c/2)
                if j % 2:
                    circle = pt.Circle((box[0]+(m-int(j/2))*width/c, box[1]+height/3), radius=1, color=color_copy[j-1])
                    ax.add_patch(circle)
                else:
                    circle = pt.Circle((box[0]+(m+int(j/2))*width/c, box[1]+height/3), radius=1, color=color_copy[j-1])
                    ax.add_patch(circle)

            for j in range(5, x_labels[i]+1):
                c = x_labels[i]-3
                m = int(c/2)
                if j % 2:
                    circle = pt.Circle((box[0]+(m-int((j-4)/2))*width/c, box[1]+2*height/3), radius=1, color=color_copy[j-1])
                    ax.add_patch(circle)
                else:
                    circle = pt.Circle((box[0]+(m+int((j-4)/2))*width/c, box[1]+2*height/3), radius=1, color=color_copy[j-1])
                    ax.add_patch(circle)
        else:
            for j in range(1, 5):
                c = 5
                m = int(c/2)
                if j % 2:
                    circle = pt.Circle((box[0]+(m-int(j/2))*width/c, box[1]+height/4), radius=1, color=color_copy[j-1])
                    ax.add_patch(circle)
                else:
                    circle = pt.Circle((box[0]+(m+int(j/2))*width/c, box[1]+height/4), radius=1, color=color_copy[j-1])
                    ax.add_patch(circle)

            for j in range(5, 9):
                c = 5
                m = int(c/2)
                if j % 2:
                    circle = pt.Circle((box[0]+(m-int((j-4)/2))*width/c, box[1]+2*height/4), radius=1, color=color_copy[j-1])
                    ax.add_patch(circle)
                else:
                    circle = pt.Circle((box[0]+(m+int((j-4)/2))*width/c, box[1]+2*height/4), radius=1, color=color_copy[j-1])
                    ax.add_patch(circle)

            for j in range(9, x_labels[i]+1):
                c = x_labels[i]-7
                m = int(c/2)
                if j % 2:
                    circle = pt.Circle((box[0]+(m-int((j-8)/2))*width/c, box[1]+3*height/4), radius=1, color=color_copy[j-1])
                    ax.add_patch(circle)
                else:
                    circle = pt.Circle((box[0]+(m+int((j-8)/2))*width/c, box[1]+3*height/4), radius=1, color=color_copy[j-1])
                    ax.add_patch(circle)




    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=30, offset=15))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax

def hue_range_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [0,1,2,3,4,5,6]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")

    # Draw Example
    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)
        if x_labels[i] == 0:
            ax.text(start+line_length/2,-10, "no different",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(start+line_length/2,-10, "low",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 4:
            ax.text(start+line_length/2,-10, "large",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "wide",  ha="center", va="center", fontsize=5)

        ranc1 = random.randint(0, 11)
        ranc2 = ranc1 + int(random.choice([1,-1])) * x_labels[i]
        if ranc2 >= 11:
            ranc2 = ranc1 - x_labels[i]
        elif ranc2 < 0:
            ranc2 = ranc1 + x_labels[i]
        ax.text(start+line_length/2,5, "e.g.", ha="center", va="center", fontsize=5)
        ax.arrow(box[0]+width/2, box[1]+height*2/3, 6, 0, width=0.25, color="k", length_includes_head=True)
        ax.arrow(box[0]+width/2, box[1]+height*2/3, -6, 0, width=0.25, color="k", length_includes_head=True)

        circle = pt.Circle((box[0]+width/5, box[1]+height*2/3), radius=3, color=color[ranc1])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*4/5, box[1]+height*2/3), radius=3, color=color[ranc2])
        ax.add_patch(circle)




    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/7, offset=360/14))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax


def sat_range_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [0,1,2,3,4,5]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")

    # Draw Example
    rgb = [0,0,0]
    rgb_choice = random.randint(0,2)
    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)
        if x_labels[i] == 0:
            ax.text(start+line_length/2,-10, "no different",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(start+line_length/2,-10, "low",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 4:
            ax.text(start+line_length/2,-10, "large",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "wide",  ha="center", va="center", fontsize=5)


        ax.text(start+line_length/2,5, "e.g.", ha="center", va="center", fontsize=5)
        ax.arrow(box[0]+width/2, box[1]+height*2/3, 6, 0, width=0.25, color="k", length_includes_head=True)
        ax.arrow(box[0]+width/2, box[1]+height*2/3, -6, 0, width=0.25, color="k", length_includes_head=True)
        circle = pt.Circle((box[0]+width/5, box[1]+height*2/3), radius=3, color=[0,0,0])
        ax.add_patch(circle)
        rgb[rgb_choice] = x_labels[i]/5
        circle = pt.Circle((box[0]+width*4/5, box[1]+height*2/3), radius=3, color=rgb)
        ax.add_patch(circle)




    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/6, offset=360/12))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax

def brt_range_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [0,1,2,3,4,5]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")

    # Draw Example
    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)
        if x_labels[i] == 0:
            ax.text(start+line_length/2,-10, "no different",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(start+line_length/2,-10, "low",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 4:
            ax.text(start+line_length/2,-10, "large",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "wide",  ha="center", va="center", fontsize=5)


        ax.text(start+line_length/2,5, "e.g.",  ha="center", va="center", fontsize=5)
        ax.arrow(box[0]+width/2, box[1]+height*2/3, 6, 0, width=0.25, color="k", length_includes_head=True)
        ax.arrow(box[0]+width/2, box[1]+height*2/3, -6, 0, width=0.25, color="k", length_includes_head=True)
        circle = pt.Circle((box[0]+width/5, box[1]+height*2/3), radius=3, ec="k", color=[0,0,0])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*4/5, box[1]+height*2/3), radius=3, ec="k", color=[x_labels[i]/5,x_labels[i]/5,x_labels[i]/5])
        ax.add_patch(circle)




    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/6, offset=360/12))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax

def sat_uni_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [1,2,3,4,5]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")

    # Draw Example
    rgb_l = []
    rgb = [0,0,0]
    rgb_choice = random.randint(0,2)
    for i in range(6):
        a = rgb.copy()
        a[rgb_choice] = i / 5
        rgb_l.append(a)
    random.shuffle(rgb_l)
    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)
        if x_labels[i] == 1:
            ax.text(start+line_length/2,-10, "mono-sat",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(start+line_length/2,-10, "few sat",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 3:
            ax.text(start+line_length/2,-10, "some sat",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 4:
            ax.text(start+line_length/2,-10, "many sat",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "various sat",  ha="center", va="center", fontsize=5)


        for j in range(1, x_labels[i]+1):
                c = x_labels[i]+1
                m = int(c/2)
                if j % 2:
                    circle = pt.Circle((box[0]+(m-int(j/2))*width/c, box[1]+height/2), radius=3, color=rgb_l[j-1])
                    ax.add_patch(circle)
                else:
                    circle = pt.Circle((box[0]+(m+int(j/2))*width/c, box[1]+height/2), radius=3, color=rgb_l[j-1])
                    ax.add_patch(circle)



    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/5, offset=360/10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax

def brt_uni_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [1,2,3,4,5]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")

    # Draw Example
    rgb_l = []

    for i in range(6):
        rgb = [i / 5, i / 5, i / 5]
        rgb_l.append(rgb)
    random.shuffle(rgb_l)
    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)
        if x_labels[i] == 1:
            ax.text(start+line_length/2,-10, "mono-brt",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(start+line_length/2,-10, "few brt",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 3:
            ax.text(start+line_length/2,-10, "some brt",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 4:
            ax.text(start+line_length/2,-10, "many brt",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "various brt",  ha="center", va="center", fontsize=5)


        for j in range(1, x_labels[i]+1):
                c = x_labels[i]+1
                m = int(c/2)
                if j % 2:
                    circle = pt.Circle((box[0]+(m-int(j/2))*width/c, box[1]+height/2), radius=3, ec="k", color=rgb_l[j-1])
                    ax.add_patch(circle)
                else:
                    circle = pt.Circle((box[0]+(m+int(j/2))*width/c, box[1]+height/2), radius=3, ec="k", color=rgb_l[j-1])
                    ax.add_patch(circle)



    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/5, offset=360/10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax

def calculate_oi():
    idx = [i for i in range(12)]
    idx_sb = [i for i in range(6)]
    def oi_cal(in_, out_, idex):
        d_in = {}
        d_out = {}
        diff = 0
        same = 0
        for i in idex:
            d_in[i] = 0
            d_out[i] = 0

        for i in in_:
            d_in[i] += 1/len(in_)
        for i in out_:
            d_out[i] += 1/len(out_)


        for i in idex:
            if d_in[i] > 0 and d_out[i] > 0:
                same += (d_in[i] + d_out[i])/2
            diff += abs(d_in[i] - d_out[i])
        diff /= 2
        diff = round(diff*10)
        if same > 0.75 and diff > 0:
            diff -= 1
        elif same < 0.25 and diff < 10:
            diff += 1
        # print(in_, out_, same, diff)

        return diff


    calculated = {}
    calculated_sb = {}

    for i in range(11):
        calculated[i] = []
        calculated_sb[i] = []

    for i in it.combinations_with_replacement(idx[:6], 6):
        for j in it.combinations_with_replacement(idx, 6):
            calculated[oi_cal(list(i), list(j), idx)].append([list(i), list(j)])

    for i in it.combinations_with_replacement(idx_sb, 6):
        for j in it.combinations_with_replacement(idx_sb, 6):
            calculated_sb[oi_cal(list(i), list(j), idx_sb)].append([list(i), list(j)])

    save_dir = "/".join(__file__.split("/")[:-1])
    with open(save_dir+"/out_in_calculated.pkl", "wb") as file:
        pickle.dump(calculated, file)
        file.close()

    with open(save_dir+"/out_in_calculated_sb.pkl", "wb") as file:
        pickle.dump(calculated_sb, file)
        file.close()



def hue_oidiff_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [0,1,2,3,4,5,6,7,8,9,10]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")

    # Draw Example

    color_copy = color.copy()
    random.shuffle(color_copy)

    calculate = calculate_hue



    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)

        if x_labels[i] == 0:
            ax.text(start+line_length/2,-10, "no different",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 1:
            ax.text(start+line_length/2,-10, "small",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(start+line_length/2,-10, "some",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 4:
            ax.text(start+line_length/2,-10, "many",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 9:
            ax.text(start+line_length/2,-10, "large",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "all different",  ha="center", va="center", fontsize=5)

        polygon = [[0.25,0.25],[0.75,0.25], [0.75,0.75], [0.25,0.75]]
        poly = pt.Polygon([[box[0]+width*j[0], box[1]+height*j[1]] for j in polygon], ec="k", fc="none", hatch="//////////")
        ax.add_patch(poly)

        example = calculate[x_labels[i]].copy()
        random.shuffle(example)
        example = example[0]
        ori = [color_copy[c] for c in example[0]]
        diff = [color_copy[c] for c in example[1]]


        circle = pt.Circle((box[0]+width*2/6, box[1]+height*2/5), radius=0.8, color=ori[0])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*3/6, box[1]+height*2/5), radius=0.8, color=ori[1])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*4/6, box[1]+height*2/5), radius=0.8, color=ori[2])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*2/6, box[1]+height*3/5), radius=0.8, color=ori[3])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*3/6, box[1]+height*3/5), radius=0.8, color=ori[4])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*4/6, box[1]+height*3/5), radius=0.8, color=ori[5])
        ax.add_patch(circle)

        circle = pt.Circle((box[0]+width*1/9, box[1]+height*1/2), radius=0.8, color=diff[0])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*8/9, box[1]+height*1/2), radius=0.8, color=diff[1])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*3/9, box[1]+height*1/9), radius=0.8, color=diff[2])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*6/9, box[1]+height*1/9), radius=0.8, color=diff[3])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*3/9, box[1]+height*8/9), radius=0.8, color=diff[4])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*6/9, box[1]+height*8/9), radius=0.8, color=diff[5])
        ax.add_patch(circle)




    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/11, offset=360/22))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax

def brt_oidiff_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [0,1,2,3,4,5,6,7,8,9,10]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")

    # Draw Example
    color_copy = [[i/5, i/5, i/5] for i in range(6)]
    random.shuffle(color_copy)

    calculate = calculate_sb

    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="royalblue")
        ax.add_patch(rect)

        if x_labels[i] == 0:
            ax.text(start+line_length/2,-10, "no different",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 1:
            ax.text(start+line_length/2,-10, "small",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(start+line_length/2,-10, "some",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 4:
            ax.text(start+line_length/2,-10, "many",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 9:
            ax.text(start+line_length/2,-10, "large",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "all different",  ha="center", va="center", fontsize=5)

        polygon = [[0.25,0.25],[0.75,0.25], [0.75,0.75], [0.25,0.75]]
        poly = pt.Polygon([[box[0]+width*j[0], box[1]+height*j[1]] for j in polygon], ec="k", fc="dodgerblue", hatch="//////////")
        ax.add_patch(poly)

        example = calculate[x_labels[i]].copy()
        random.shuffle(example)
        example = example[0]
        ori = [color_copy[c] for c in example[0]]
        diff = [color_copy[c] for c in example[1]]


        circle = pt.Circle((box[0]+width*2/6, box[1]+height*2/5), radius=0.8, color=ori[0])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*3/6, box[1]+height*2/5), radius=0.8, color=ori[1])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*4/6, box[1]+height*2/5), radius=0.8, color=ori[2])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*2/6, box[1]+height*3/5), radius=0.8, color=ori[3])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*3/6, box[1]+height*3/5), radius=0.8, color=ori[4])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*4/6, box[1]+height*3/5), radius=0.8, color=ori[5])
        ax.add_patch(circle)

        circle = pt.Circle((box[0]+width*1/9, box[1]+height*1/2), radius=0.8, color=diff[0])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*8/9, box[1]+height*1/2), radius=0.8, color=diff[1])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*3/9, box[1]+height*1/9), radius=0.8, color=diff[2])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*6/9, box[1]+height*1/9), radius=0.8, color=diff[3])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*3/9, box[1]+height*8/9), radius=0.8, color=diff[4])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*6/9, box[1]+height*8/9), radius=0.8, color=diff[5])
        ax.add_patch(circle)




    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/11, offset=360/22))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax

def sat_oidiff_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [0,1,2,3,4,5,6,7,8,9,10]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")

    # Draw Example
    color_copy = []
    rgb = [0,0,0]
    rgb_choice = random.randint(0,2)
    for i in range(6):
        a = rgb.copy()
        a[rgb_choice] = i/5
        color_copy.append(a)

    random.shuffle(color_copy)

    calculate = calculate_sb

    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)

        if x_labels[i] == 0:
            ax.text(start+line_length/2,-10, "no different",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 1:
            ax.text(start+line_length/2,-10, "small",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(start+line_length/2,-10, "some",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 4:
            ax.text(start+line_length/2,-10, "many",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 9:
            ax.text(start+line_length/2,-10, "large",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "all different",  ha="center", va="center", fontsize=5)

        polygon = [[0.25,0.25],[0.75,0.25], [0.75,0.75], [0.25,0.75]]
        poly = pt.Polygon([[box[0]+width*j[0], box[1]+height*j[1]] for j in polygon], ec="k", fc="none", hatch="//////////")
        ax.add_patch(poly)

        example = calculate[x_labels[i]].copy()
        random.shuffle(example)
        example = example[0]
        ori = [color_copy[c] for c in example[0]]
        diff = [color_copy[c] for c in example[1]]


        circle = pt.Circle((box[0]+width*2/6, box[1]+height*2/5), radius=0.8, color=ori[0])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*3/6, box[1]+height*2/5), radius=0.8, color=ori[1])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*4/6, box[1]+height*2/5), radius=0.8, color=ori[2])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*2/6, box[1]+height*3/5), radius=0.8, color=ori[3])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*3/6, box[1]+height*3/5), radius=0.8, color=ori[4])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*4/6, box[1]+height*3/5), radius=0.8, color=ori[5])
        ax.add_patch(circle)

        circle = pt.Circle((box[0]+width*1/9, box[1]+height*1/2), radius=0.8, color=diff[0])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*8/9, box[1]+height*1/2), radius=0.8, color=diff[1])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*3/9, box[1]+height*1/9), radius=0.8, color=diff[2])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*6/9, box[1]+height*1/9), radius=0.8, color=diff[3])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*3/9, box[1]+height*8/9), radius=0.8, color=diff[4])
        ax.add_patch(circle)
        circle = pt.Circle((box[0]+width*6/9, box[1]+height*8/9), radius=0.8, color=diff[5])
        ax.add_patch(circle)



    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/11, offset=360/22))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax

def size_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [0,1,2,3,4]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")

    # Draw Example

    poly_l = random_star_shaped_polygon(num_points=12)


    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)

        if x_labels[i] == 0:
            ax.text(start+line_length/2,-10, "really small",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 1:
            ax.text(start+line_length/2,-10, "small",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(start+line_length/2,-10, "neutral",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 3:
            ax.text(start+line_length/2,-10, "large",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "really large",  ha="center", va="center", fontsize=5)

        poly = []
        for c in poly_l:
            poly.append([box[0] + c[0]*width*((i+1)/6) + width*((5-i)/6)/2, box[1] + c[1]*height*((i+1)/6) + height*((5-i)/6)/2])

        d_area = pt.Polygon(poly, color="w", ec="k", hatch="//////////")
        ax.add_patch(d_area)



    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/5, offset=360/10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax

def asp_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [0,1,2,3,4]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")

    # Draw Example
    poly_l = random_star_shaped_polygon(num_points=12)


    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)

        if x_labels[i] == 0:
            ax.text(start+line_length/2,-10, "really small/simple",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 1:
            ax.text(start+line_length/2,-10, "small/simple",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(start+line_length/2,-10, "large/complex",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "really large/complex",  ha="center", va="center", fontsize=5)


        w = height* (i+2)/3
        h = height*(2/3)

        sw = box[0] + (width-w)/2
        sh = box[1] + height*(1/3)/2

        d_box = pt.Rectangle((sw,sh), width=w, height=h, color="none", ec="k",  linestyle=":")
        ax.add_patch(d_box)
        poly = []
        for c in poly_l:
            poly.append([sw+c[0]*w, sh+c[1]*h])
        d_area = pt.Polygon(poly, color="none", ec="k", hatch="//////////")
        ax.add_patch(d_area)



    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/5, offset=360/10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax

def cov_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [0,1,2,3,4,5]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")

    # Draw Example

    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)

        if x_labels[i] == 0:
            ax.text(start+line_length/2,-10, "empty 0-10%",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 1:
            ax.text(start+line_length/2,-10, "smaller 10-30%",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(start+line_length/2,-10, "small 30-50%",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 3:
            ax.text(start+line_length/2,-10, "large 50-70%",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 4:
            ax.text(start+line_length/2,-10, "larger 70-90%",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "full 90-100%",  ha="center", va="center", fontsize=5)


        w = width * (2/3)
        h = height * (2/3)

        sw = box[0] + (width-w)/2
        sh = box[1] + (height-h)/2

        d_box = pt.Rectangle((sw,sh), width=w, height=h, color="none", ec="k",  linestyle=":")
        ax.add_patch(d_box)
        poly_l = []
        if x_labels[i] == 0:
            poly_l = [[1.0, 1.0], [0.45, 0.25], [0, 0], [0.45, 0.45]]
        elif x_labels[i] <= 1:
            poly_l = [[1.0, 1.0], [0.45, 0.25], [0, 0], [0.45, 0.8]]
        elif x_labels[i] <= 2:
            poly_l = [[1.0, 1.0], [0, 1.0], [0, 0], [0.5,0.7]]
        elif x_labels[i] <= 3:
            poly_l = [[1.0, 1.0], [0, 1.0], [0, 0], [0.5,0.3]]
        elif x_labels[i] <= 4:
            poly_l = [[1.0, 1.0], [0, 1.0], [0, 0], [0.5, 0], [1.0, 0.5]]
        else:
            poly_l = [[1.0, 1.0], [0, 1.0], [0, 0], [1.0, 0]]

        poly = []
        for c in poly_l:
            poly.append([sw+c[0]*w, sh+c[1]*h])
        d_area = pt.Polygon(poly, color="none", ec="k", hatch="//////////")
        ax.add_patch(d_area)



    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/6, offset=360/12))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax

def edge_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [0,1,2,3,4]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")


    # Draw Example
    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)

        if x_labels[i] == 0:
            ax.text(start+line_length/2,-10, "few",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 1:
            ax.text(start+line_length/2,-10, "some",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 3:
            ax.text(start+line_length/2,-10, "many",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "too much",  ha="center", va="center", fontsize=5)


        w = width * (4/5)
        h = height * (4/5)

        sw = box[0] + (width-w)/2
        sh = box[1] + (height-h)/2

        poly_l = random_star_shaped_polygon(num_points=(x_labels[i]+1)*5)

        poly = []
        for c in poly_l:
            poly.append([sw+c[0]*w, sh+c[1]*h])


        d_area = pt.Polygon(poly, color="none", ec="k", hatch="//////////")
        ax.add_patch(d_area)



    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/5, offset=360/10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax

def dist_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [0,1,2]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")



    # Draw Example
    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)

        if x_labels[i] == 0:
            ax.text(start+line_length/2,-10, "short",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 1:
            ax.text(start+line_length/2,-10, "long",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "no neighbours",  ha="center", va="center", fontsize=5)

        color_copy = color.copy()
        random.shuffle(color_copy)

        if x_labels[i] == 0:
            d_area = pt.Circle([box[0]+width*2/5, box[1]+height/2], radius=5, color=color_copy[0], ec="k", hatch="//////////")
            ax.add_patch(d_area)
            d_area = pt.Circle([box[0]+width*3/5, box[1]+height/2], radius=5, color=color_copy[1], ec="k", hatch="//////////")
            ax.add_patch(d_area)
        elif x_labels[i] <= 1:
            d_area = pt.Circle([box[0]+width*1/5, box[1]+height/2], radius=5, color=color_copy[0], ec="k", hatch="//////////")
            ax.add_patch(d_area)
            d_area = pt.Circle([box[0]+width*4/5, box[1]+height/2], radius=5, color=color_copy[1], ec="k", hatch="//////////")
            ax.add_patch(d_area)
        else:
            d_area = pt.Circle([box[0]+width/2, box[1]+height/2], radius=5, color=color_copy[0], ec="k", hatch="//////////")
            ax.add_patch(d_area)




    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/3, offset=360/6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax


def deg_avg_md_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [0,1,2,3,4,5,6]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")



    # Draw Example
    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)

        if x_labels[i] == 0:
            ax.text(start+line_length/2,-10, "sharp angle",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 1:
            ax.text(start+line_length/2,-10, "small acute angle",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(start+line_length/2,-10, "acute angle",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 3:
            ax.text(start+line_length/2,-10, "right angle",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 4:
            ax.text(start+line_length/2,-10, "obtuse angle",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 5:
            ax.text(start+line_length/2,-10, "large obtuse angle",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "straight line",  ha="center", va="center", fontsize=5)


        if x_labels[i] == 0:
            ax.plot([box[0]+width/4, box[0]+width*3/4], [box[1]+height/2, box[1]+height*4/10], c="k",  linestyle="-")
            ax.plot([box[0]+width/4, box[0]+width*3/4], [box[1]+height/2, box[1]+height*6/10], c="k",  linestyle="-")
        elif x_labels[i] <= 1:
            ax.plot([box[0]+width/4, box[0]+width*3/4], [box[1]+height/2, box[1]+height*2/10], c="k",  linestyle="-")
            ax.plot([box[0]+width/4, box[0]+width*3/4], [box[1]+height/2, box[1]+height*8/10], c="k",  linestyle="-")
        elif x_labels[i] <= 2:
            ax.plot([box[0]+width/3, box[0]+width*2/3], [box[1]+height*3/4, box[1]+height*3/4], c="k",  linestyle="-")
            ax.plot([box[0]+width/3, box[0]+width*2/4], [box[1]+height*3/4, box[1]+height*1/8], c="k",  linestyle="-")
        elif x_labels[i] <= 3:
            ax.plot([box[0]+width/3, box[0]+width*2/3], [box[1]+height*3/4, box[1]+height*3/4], c="k",  linestyle="-")
            ax.plot([box[0]+width/3, box[0]+width/3], [box[1]+height*3/4, box[1]+height/8], c="k",  linestyle="-")
        elif x_labels[i] <= 4:
            ax.plot([box[0]+width/2, box[0]+width*4/5], [box[1]+height*3/4, box[1]+height*3/4], c="k",  linestyle="-")
            ax.plot([box[0]+width/2, box[0]+width*2/5], [box[1]+height*3/4, box[1]+height/8], c="k",  linestyle="-")
        elif x_labels[i] <= 5:
            ax.plot([box[0]+width/2, box[0]+width*4/5], [box[1]+height*3/4, box[1]+height*3/4], c="k",  linestyle="-")
            ax.plot([box[0]+width/2, box[0]+width*1/5], [box[1]+height*3/4, box[1]+height*3/8], c="k",  linestyle="-")
        else:
            ax.plot([box[0]+width/2, box[0]+width*1/4], [box[1]+height/2, box[1]+height/2], c="k",  linestyle="-")
            ax.plot([box[0]+width/2, box[0]+width*3/4], [box[1]+height/2, box[1]+height/2], c="k",  linestyle="-")




    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/7, offset=360/14))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax

def er_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [0,1,2,3,4]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")



    # Draw Example
    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)

        if x_labels[i] == 0:
            ax.text(start+line_length/2,-10, "really small/complex",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 1:
            ax.text(start+line_length/2,-10, "small/complex",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(start+line_length/2,-10, "neutral",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 3:
            ax.text(start+line_length/2,-10, "large/simple",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "really large/simple",  ha="center", va="center", fontsize=5)


        if x_labels[i] == 0:
            ax.plot([box[0]+width/5, box[0]+width*2/7], [box[1]+height/2, box[1]+height*2/6], c="k",  linestyle="-")
            ax.plot([box[0]+width/5, box[0]+width*4/5], [box[1]+height/2, box[1]+height*5/6], c="k",  linestyle="-")
        elif x_labels[i] <= 1:
            ax.plot([box[0]+width/5, box[0]+width*3/7], [box[1]+height/2, box[1]+height*2/6], c="k",  linestyle="-")
            ax.plot([box[0]+width/5, box[0]+width*4/5], [box[1]+height/2, box[1]+height*5/6], c="k",  linestyle="-")
        elif x_labels[i] <= 2:
            ax.plot([box[0]+width/5, box[0]+width*1/2], [box[1]+height/2, box[1]+height*2/6], c="k",  linestyle="-")
            ax.plot([box[0]+width/5, box[0]+width*4/5], [box[1]+height/2, box[1]+height*5/6], c="k",  linestyle="-")
        elif x_labels[i] <= 3:
            ax.plot([box[0]+width/5, box[0]+width*3/5], [box[1]+height/2, box[1]+height*1/6], c="k",  linestyle="-")
            ax.plot([box[0]+width/5, box[0]+width*4/5], [box[1]+height/2, box[1]+height*5/6], c="k",  linestyle="-")
        else:
            ax.plot([box[0]+width/5, box[0]+width*4/5], [box[1]+height/2, box[1]+height*1/6], c="k",  linestyle="-")
            ax.plot([box[0]+width/5, box[0]+width*4/5], [box[1]+height/2, box[1]+height*5/6], c="k",  linestyle="-")







    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/5, offset=360/10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax

def rt_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [0,1,2,3,4]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")





    # Draw Example
    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)

        if x_labels[i] == 0:
            ax.text(start+line_length/2,-10, "none/simple",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 1:
            ax.text(start+line_length/2,-10, "few/simple",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(start+line_length/2,-10, "some",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 3:
            ax.text(start+line_length/2,-10, "many/complex",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "more/complex",  ha="center", va="center", fontsize=5)

        w = width * (4/5)
        h = height * (4/5)

        sw = box[0] + (width-w)/2
        sh = box[1] + (height-h)/2
        poly_l = []

        poly_x = []
        poly_y = []

        if x_labels[i] == 0:
            poly_l = random_convex_polygon(num_points=5)
        elif x_labels[i] == 1:
            poly_l = [[0.2,0.3], [0.6, 0.1], [0.95, 0.45],[0.7,0.7], [0.9, 0.85], [0.5, 0.9], [0.3, 0.75]]
        elif x_labels[i] == 2:
            poly_l = [[0.15,0.15], [0.3,0.3], [0.6,0.05], [0.8,0.25], [0.95,0.6],[0.98,0.3],[1, 0.95],[0.75,0.6],[0.5, 0.6],[0.45,1], [0.3,0.45],[0.1,0.6],[0.05,0.2] ]
        elif x_labels[i] == 3:
            poly_l = [[0.2,0.3], [0.6,0.1], [0.4,0.4], [0.8,0.2],[0.9,0.6],[0.95,0.35],[0.93, 0.85],[0.8,0.95],[0.6,0.6],[0.6,0.95],[0.4,0.55],[0.25,0.9], [0.1, 0.3]]
        else:
            poly_l = [[0.2,0.3], [0.4, 0.1], [0.55, 0.3],[0.95,0.05], [0.25, 0.9], [0.7, 0.95]]


        for c in poly_l:
            poly_x.append(sw+c[0]*w)
            poly_y.append(sh+c[1]*h)
        ax.plot(poly_x[:-1], poly_y[:-1], c="k",  linestyle="-")
        ax.plot(poly_x[-2:], poly_y[-2:], c="k",  linestyle=":")




    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/5, offset=360/10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax


def ft_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [0,1,2,3,4]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")





    # Draw Example
    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)

        if x_labels[i] == 0:
            ax.text(start+line_length/2,-10, "none/complex",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 1:
            ax.text(start+line_length/2,-10, "few/complex",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(start+line_length/2,-10, "some",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 3:
            ax.text(start+line_length/2,-10, "many/simple",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "more/simple",  ha="center", va="center", fontsize=5)

        w = width * (4/5)
        h = height * (4/5)

        sw = box[0] + (width-w)/2
        sh = box[1] + (height-h)/2
        poly_l = []

        poly=[]

        if x_labels[i] == 4:
            poly_l = [[0,0],[1,0],[1,1],[0,1]]
        elif x_labels[i] == 3:
            poly_l = random_convex_polygon(num_points=8)
        elif x_labels[i] == 2:
            poly_l = [[0.2,0.3], [0.6,0.05], [0.8,0.15], [0.95,0.6],[0.98,0.3],[1, 0.95],[0.75,0.6],[0.5, 0.6],[0.45,1], [0.3,0.45],[0.1,0.6],[0.1,0] ]
        elif x_labels[i] == 1:
            poly_l = [[0.2,0.3], [0.6,0.1], [0.4,0.4], [0.6,0.4], [0.7,0.1],[0.8,0.4], [0.85,0.2],[0.9,0.6],[0.95,0.35],[0.93, 0.85],[0.8,0.95],[0.6,0.6],[0.6,0.95],[0.4,0.55],[0.25,0.9], [0.1, 0.3]]
        else:
            poly_l = [(0.8793401801908968, 0.3484071706122948), (0.9999999999999999, 0.3530058933327532), (0.5836088485778805, 1.0), (0.8716833681646179, 0.3963895537990236), (0.5891293718637959, 0.4476825974677554), (0.43071289684226205, 0.32165426091941424), (0.31685006981747316, 0.2249860420195398), (0.5045252028564648, 0.9272282750186512), (0.3699352041908622, 0.9291346031627217), (0.043993538634781705, 0.5448918287795448), (0.09816764475344923, 0.45112687256194123), (0.0, 0.0), (0.6840340732959219, 0.0021952894516045927), (0.16921979615182592, 0.11859241417260867), (0.09889255788394617, 0.15236536723152375)]


        for c in poly_l:
            poly.append([sw+c[0]*w, sh+c[1]*h])

        d_area = pt.Polygon(poly, color="none", ec="k", hatch="//////////")
        ax.add_patch(d_area)




    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/5, offset=360/10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax

def st_bar_plot(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):

    colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = [0,1,2,3,4]


    # Calculate Bound
    left = rang[0] * 359
    right = rang[1] * 359
    left_s = range_stable[0] * 359
    right_s = range_stable[1] * 359
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i/359 for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=colour_choice)

    # Draw Mark Line
    line_length = 360 / (len(x_labels))
    gap = 0.025 * line_length
    for i in range(len(x_labels)):
        if i == 0:
            ax.plot([0, line_length-gap] , [22, 22], c="k")
        elif i == len(x_labels)-1:
            ax.plot([360-(line_length-gap), 359] , [22, 22], c="k")
        else:
            start = (i) * line_length + gap
            end = (i+1) * line_length - gap
            ax.plot([start, end] , [22, 22], c="k")





    # Draw Example
    for i in range(len(x_labels)):
        width = 360 / len(x_labels) * 0.8
        height = 20 * 0.7
        start = i*line_length
        box = [start+(line_length-width)/2, (20-height)/2]
        rect = pt.Rectangle((start+(line_length-width)/2,(20-height)/2), width, height, color="w", ec="k")
        ax.add_patch(rect)

        if x_labels[i] == 0:
            ax.text(start+line_length/2,-10, "none/simple",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 1:
            ax.text(start+line_length/2,-10, "few/simple",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 2:
            ax.text(start+line_length/2,-10, "some",  ha="center", va="center", fontsize=5)
        elif x_labels[i] <= 3:
            ax.text(start+line_length/2,-10, "many/complex",  ha="center", va="center", fontsize=5)
        else:
            ax.text(start+line_length/2,-10, "more/complex",  ha="center", va="center", fontsize=5)

        w = width * (4/5)
        h = height * (4/5)

        sw = box[0] + (width-w)/2
        sh = box[1] + (height-h)/2
        poly_l = []

        poly=[]

        if x_labels[i] == 4:
            poly_l = random_polygon(num_points=30)
        elif x_labels[i] == 3:
            poly_l = random_star_shaped_polygon(num_points=25)
        elif x_labels[i] == 2:
            poly_l = random_star_shaped_polygon(num_points=12)
        elif x_labels[i] == 1:
            poly_l = random_convex_polygon(num_points=8)
        else:
            poly_l = [[0,0.25],[0.25,0],[0.75,0],[1,0.25],[1,0.75],[0.75,1],[0.25,1],[0,0.75]]

        for c in poly_l:
            poly.append([sw+c[0]*w, sh+c[1]*h])

        d_area = pt.Polygon(poly, color="none", ec="k", hatch="//////////")
        ax.add_patch(d_area)




    # Left Bound
    ax.plot([left,left], [-1,20], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k",  linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k",  linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k",  linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k",  linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k",  linestyle="-")
        ax.plot([359,354], [-1,-1], c="k",  linestyle="-")
        ax.plot([359,354], [20,20], c="k",  linestyle="-")

        ax.plot([0,0], [-1,20], c="k",  linestyle="-")
        ax.plot([0,5], [-1,-1], c="k",  linestyle="-")
        ax.plot([0,5], [20,20], c="k",  linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=0.5, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=0.5, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "DOS: " + str(overall) + " | "
    if score is not None:
        Title += "DIS: " + str(score) + " | "
    if count is not None:
        Title += "DUF: " + str(count)

    ax.set_title(Title, fontweight="bold", y=1.8)

    # Setting
    # ax.plot([-1,360], [-3,-3], visible=False)
    # ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=360/5, offset=360/10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', colors="k")


    return ax


def explain_forest(report, feature, folder_path, route=None, top_route=5, range_split=True, detail=True):

    realpath = os.path.realpath(folder_path)+"/"
    if not os.path.exists(realpath):
        logging.warning("Directory is not found, Create one.")
        os.mkdir(realpath)

    logging.info("Plot Summary")
    fig = summary_plot(report, feature)
    fig.savefig(realpath+"summary.jpg", bbox_inches="tight")

    logging.info("Plot Range Summary")
    if range_split:
        fig = summary_range_plot(report, feature, detail=False)
        fig.savefig(realpath + "summary_range.jpg", bbox_inches="tight")

        sort_report = sorted([report[i] for i in feature], key=lambda x: x["overall"], reverse=True)
        count = 1
        for i in sort_report:
            fig = plt.Figure(figsize=(2000 / 300, 500 / 300), dpi=300)
            ax = fig.add_axes([0, 0, 1, 1])
            ax = plot_dict[i["feature"]](ax, i["range"], i["range_stable"], dc_name=feature_name[i["feature"]],
                                         overall=str(round(i["overall_norm"] * 100)) + "%",
                                         score=str(round(i["score_norm"] * 100)) + "%", count=str(round(i["count"], 2)))
            count += 1
            fig.savefig(realpath + "range_" + feature_name[i["feature"]] + ".jpg", bbox_inches="tight")

    else:
        fig = summary_range_plot(report, feature, detail=detail)
        fig.savefig(realpath + "summary_range.jpg", bbox_inches="tight")


    logging.info("Plot Top Routes")
    if route is not None:
        for i in range(top_route):
            fig = specific_route_plot(route, i, detail=detail)
            fig.savefig(realpath + "route_value_top"+str(i)+".jpg", bbox_inches="tight")
            fig = specific_route_plot(route, i, key="length", rev=False, detail=detail)
            fig.savefig(realpath + "route_length(short)_top" + str(i) + ".jpg", bbox_inches="tight")
            fig = specific_route_plot(route, i, key="score", detail=detail)
            fig.savefig(realpath + "route_score_top" + str(i) + ".jpg", bbox_inches="tight")

    output_text = "This section will explain the possible processes for improving the detection " \
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
    output_text += "Number of Unique Saturation:\n1. Normalise colour\n2. Improve detection model architecture\n3. Grey-scale " \
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
    with open(realpath+"improvement_recommendations.txt", "w") as txt_file:
        txt_file.write(output_text)
        txt_file.close()

def single_plot(data, feature, folder_path):
    realpath = os.path.realpath(folder_path) + "/"
    if not os.path.exists(realpath):
        logging.warning("Directory is not found, Create one.")
        os.mkdir(realpath)

    for i in feature:
        fig = plt.Figure(figsize=(2000 / 300, 500 / 300), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        ax = plot_dict[i](ax, [data[i], data[i]], [0,0], dc_name=i)

        fig.savefig(realpath + i + ".jpg", bbox_inches="tight")

file_path = "/".join(__file__.split("/")[:-1]) + "/out_in_calculated.pkl"

with open(file_path, "rb") as file:
    calculate_hue = pickle.load(file)
    file.close()

file_path = "/".join(__file__.split("/")[:-1]) + "/out_in_calculated_sb.pkl"

with open(file_path, "rb") as file:
    calculate_sb = pickle.load(file)
    file.close()

plot_dict = {'out_hue_avg': hue_avg_md_bar_plot,
                    'out_hue_mode': hue_avg_md_bar_plot,
                    'out_hue_range': hue_range_bar_plot,
                    'out_hue_uni': hue_uni_bar_plot,
                    'out_sat_avg': sat_avg_md_bar_plot,
                    'out_sat_mode': sat_avg_md_bar_plot,
                    'out_sat_range': sat_range_bar_plot,
                    'out_sat_uni': sat_uni_bar_plot,
                    'out_brt_avg': brt_avg_md_bar_plot,
                    'out_brt_mode': brt_avg_md_bar_plot,
                    'out_brt_range': brt_range_bar_plot,
                    'out_brt_uni': brt_uni_bar_plot,
                    'size': size_bar_plot,
                    'asp_ratio': asp_bar_plot,
                    'distance': dist_bar_plot,
                    'coverage': cov_bar_plot,
                    'deg_avg': deg_avg_md_bar_plot,
                    'deg_mode': deg_avg_md_bar_plot,
                    'edge': edge_bar_plot,
                    'sc_edge_ratio': er_bar_plot,
                    'sc_follow_turn': ft_bar_plot,
                    'sc_reverse_turn': rt_bar_plot,
                    'sc_small_turn': st_bar_plot,
                    'hue_avg': hue_avg_md_bar_plot,
                    'hue_mode': hue_avg_md_bar_plot,
                    'hue_range': hue_range_bar_plot,
                    'hue_uni': hue_uni_bar_plot,
                    'sat_avg': sat_avg_md_bar_plot,
                    'sat_mode': sat_avg_md_bar_plot,
                    'sat_range': sat_range_bar_plot,
                    'sat_uni': sat_uni_bar_plot,
                    'brt_avg': brt_avg_md_bar_plot,
                    'brt_mode': brt_avg_md_bar_plot,
                    'brt_range': brt_range_bar_plot,
                    'brt_uni': brt_uni_bar_plot,
                    'hue_outin': hue_oidiff_bar_plot,
                    'sat_outin': sat_oidiff_bar_plot,
                    'brt_outin': brt_oidiff_bar_plot}