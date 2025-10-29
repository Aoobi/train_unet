from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import cv2
import os
import shutil
import glob

def find_fiber_and_crop(model, img_path, crop_size):
    new_folder_name = 'fiber_not_detected'
    directory = os.path.dirname(img_path)
    new_folder_path = os.path.join(directory, new_folder_name)
    res = model(img_path)
    try:
        box_data = res[0].boxes.xyxy[0]
        img = cv2.imread(img_path, 0)
        try:
            box_data.shape[1]
            labels = res[0].boxes.cls[0]
            img_crop = img[
                       int(labels[3] / 2 + labels[1] / 2) - int(crop_size / 2):int(labels[3] / 2 + labels[1] / 2) + int(
                           crop_size / 2),
                       int(labels[2] / 2 + labels[0] / 2) - int(crop_size / 2):int(labels[2] / 2 + labels[0] / 2) + int(
                           crop_size / 2)]
            cv2.imwrite(img_path.replace('.tif', '.png'), img_crop)
        except:
            try:
                labels = box_data
                img_crop = img[
                           int(labels[3] / 2 + labels[1] / 2) - int(crop_size / 2):int(labels[3] / 2 + labels[1] / 2) + int(
                               crop_size / 2),
                           int(labels[2] / 2 + labels[0] / 2) - int(crop_size / 2):int(labels[2] / 2 + labels[0] / 2) + int(
                               crop_size / 2)]
                cv2.imwrite(img_path.replace('.tif', '.png'), img_crop)
            except Exception as e:
                raise Exception
    except:
        try:
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            new_file_path = os.path.join(new_folder_path, os.path.basename(img_path))
            shutil.move(img_path, new_file_path)
        except Exception as e:
            print(e)

def locate_cut_main(folder_path):
    model8 = YOLO(r"models\fibers_detection_yolov8n.pt")
    for file in glob.glob(os.path.join(folder_path, "*.png")):
        find_fiber_and_crop(model8, file, 768)

if __name__ == '__main__':
    pass
    # locate_cut_main(r"C:\Users\pawlowskj\Downloads\1_12_V&G_Euclid\1_12_V&G_Euclid\V&G_results")

    # model8 = YOLO(r"models\fibers_detection_yolov8n.pt")
    # for file in glob.glob(os.path.join(r"C:\Users\pawlowskj\Downloads\1_12_V&G_Euclid_test_new_repo\1_12_V&G_Euclid\V&G_results", "*.png")):
    #     pass
    #
    # res = model8(file)
    # box_data = res[0].boxes.xyxy[0]
    # res[0].boxes.cls[0]
