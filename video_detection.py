from __future__ import division

from models import *
from utils.utils import *
from detection_tools.detect_uilts import *
import os
import sys
import time
import datetime
import argparse
import cv2
import torch
from detection_tools.display_utils import open_window, set_display, show_fps, BBoxVisualization
from detection_tools.detect_uilts import add_camera_args, Detect
from detection_tools.utils import *
from torch.autograd import Variable
import torchvision.transforms as transforms
from utils.datasets import *


WINDOW_NAME = 'Video_detection'


def changeBGR2RGB(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b

    return img


def changeRGB2BGR(img):
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r

    return img




def parser_args():
    parser = argparse.ArgumentParser()
    parser = add_camera_args(parser)
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    # parser.add_argument("--video_file", type=str, default="vedio_samples/video-01.mp4", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/ptsc.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="model_trained/ptsc-new-50-epoch.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/ptsc.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.07, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=3, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(args.model_def).to(device)
    if args.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(args.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(args.weights_path))
    model.eval()  # Set in evaluation mode
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_size = (args.img_size, args.img_size)
    classes = load_classes(args.class_path)
    cam = Detect(args)

    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    # open_window(
    #     WINDOW_NAME, 'Video_detecion',
    #     cam.img_width, cam.img_height)

    detections = [ ]
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")


    fps = 0.0
    tic = time.time()
    # loop and detect
    while True :
        img = cam.read()  # 비디오로부터 프레임 반환
        if img is None:
            break
        # img -> ndarray format
        # resized = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (854, 480), interpolation=cv2.INTER_CUBIC) # 이미지 resize

        RGBimg=changeBGR2RGB(img)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))


        # boxes = model(imgTensor) # get deteection result
        with torch.no_grad():
            det = model(imgTensor)
            det = non_max_suppression(det, conf_thres=args.conf_thres, iou_thres=args.nms_thres)
        detections.clear()

        if det is not None:
            detections.extend(det) # extend

        if len(detections):
            for detected_result in detections:
                detected_result = rescale_boxes(detected_result, args.img_size, img.shape[:2])
                unique_labels = detected_result[:,-1].cpu().unique()
                n_cls_preds = len(unique_labels)

                for x1, y1, x2, y2, conf ,cls_pred in detected_result:
                    print(x2)
                    print(type(x2))
                    box_w = x2 - x1
                    box_h = y2 - y1
                    color = [int(c) for c in colors[int(cls_pred)]]
                    # print(cls_conf)
                    img = cv2.rectangle(img, (int(x1), int(y1 + box_h)), (int(x2), int(y1)), color, 2)
                    cv2.putText(img, classes[int(cls_pred)], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(img, str("%.2f" % float(conf)), (int(x2), int(y2 - box_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)


        cv2.imshow('frame', changeRGB2BGR(RGBimg))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()