import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pickle


def Test_Result_Import(Label, Result_Image_Dir, specific_detector=None, reload=None, save=None):

    test = {"detection":{}, "type":{}}
    if reload is not None:
        with open(os.path.abspath(reload), 'rb') as f:
            test = pickle.load(f)
            f.close()

        return test
    image_file = []
    for r, d, f, in os.walk(Result_Image_Dir):
        image_file = f
    # c = 1
    for i in Label:
    #     if c == 323:
    #         print(Label[i]["filename"])
    #         break
    #     else:
    #         c += 1
    #         continue
        box = Label[i]["boundary"]
        image = cv2.imread(os.path.abspath(Result_Image_Dir) + "/" + Label[i]["filename"])
        adj_box = box[0]
        if specific_detector == "Mask R-CNN":
            # Y, X
            start_point = [1624, 1816]
            # W, H
            adj_shape = [0, 0]
            if box[-1][0] >= box[-1][1]:
                start_point[0] = int(1624 + (9478 - (9478 / box[-1][0] * box[-1][1])) / 2)
                adj_shape = [9478, int(9478 / box[-1][0] * box[-1][1])]
            else:
                start_point[1] = int(1816 + (9478 - (9478 / box[-1][1] * box[-1][0])) / 2)
                adj_shape = [int(9478 / box[-1][1] * box[-1][0]), 9478]

            adj_box = [int(start_point[1] + adj_box[0] * adj_shape[0] / box[-1][0]),
                       int(start_point[0] + adj_box[1] * adj_shape[1] / box[-1][1]),
                       int(start_point[1] + adj_box[2] * adj_shape[0] / box[-1][0]),
                       int(start_point[0] + adj_box[3] * adj_shape[1] / box[-1][1])]

        image = cv2.rectangle(image, (adj_box[0], adj_box[1]), (adj_box[2], adj_box[3]), (255,0,0), 50)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        show_img = Image.fromarray(image).resize((800, 800))
        plt.imshow(show_img)
        plt.show()
        detection = int(input("Detected?"))
        type = int(input("Type Corrected?"))
        test["detection"][i] = detection
        test["type"][i] = type
        plt.close()

    if save is not None:
        with open(os.path.abspath(save), 'wb') as f:
            pickle.dump(test, f)
            f.close()

    return test

def Test_Result_Import_correct(Label, Result_Image_Dir, specific_detector=None, reload=None, save=None):

    test = {"detection":{}, "type":{}}
    if reload is not None:
        with open(os.path.abspath(reload), 'rb') as f:
            test = pickle.load(f)
            f.close()

    test2 = {"detection":{}, "type":{}}
    for i in test["detection"]:
        print(i)
        if str(i).startswith("190"):
            a = input()
            if a == "-1":
                test2["detection"][i] = test["detection"][i]
                test2["type"][i] = test["type"][i]
                print(i)
            elif a == "0":
                print("skip")
                continue
            elif a == "":
                j = "_".join(str(i).split("."))
                test2["detection"][j] = test["detection"][i]
                test2["type"][j] = test["type"][i]
                print(j)
            else:
                test2["detection"][a] = test["detection"][i]
                test2["type"][a] = test["type"][i]
                print(a)
        elif str(i).startswith("191"):
            if i == 191.1:
                j = "_".join(str(i).split("."))
                test2["detection"][j] = 1
                test2["type"][j] = 1
            else:
                print("delete ", i)
                continue
        else:
            j = "_".join(str(i).split("."))
            test2["detection"][j] = test["detection"][i]
            test2["type"][j] = test["type"][i]
            print(j)
        print("===")


    if save is not None:
        with open(os.path.abspath(save), 'wb') as f:
            pickle.dump(test2, f)
            f.close()

    return test


