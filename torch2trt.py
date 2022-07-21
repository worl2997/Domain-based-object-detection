import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime
import torch
from models import Darknet

def parse_args():
    desc = ('model transfromation pytorch -> onnx -> Tensorrt')
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '-b', '--batch', type=int, default=1,
        help='batch size')
    parser.add_argument(
        '-i', '--img_size', type=int, default=416,
        help='set the input image size ')
    parser.add_argument(
        '-w', '--weight', type=str, required=True,
        help='set the model weight path')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='set the cfg file path ')
    parser.add_argument(
        '-s', '--workspace', type=int, default=4096,
        help='set the workspace size for TRT trasnformation')
    args = parser.parse_args()
    return args

def transform_to_trt(pre_weights,model_cfg, batch_size, IN_IMAGE_SIZE, space_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(model_cfg).to(device)
    file_name = model_cfg.split('/')[-1].split('.')[0]
    trt_save_name = file_name +  ".engine"

    # If pretrained weights are specified, start from checkpoint or weight file
    if pre_weights:
        if pre_weights.endswith(".pt"):
            # Load checkpoint weights
            print('pre_wegiht:',pre_weights)
            model.load_state_dict(torch.load(pre_weights, map_location=device))
            print('load model ')
        else:
            # Load darknet weights
            model.load_darknet_weights(pre_weights)

    input_names = ["input"]
    output_names = ["outputs"]
    dynamic = False  # for support dynamic batch, not supported yet



    if batch_size <= 0:
        dynamic = True

    if dynamic:
        x = torch.randn((1, 3, IN_IMAGE_SIZE, IN_IMAGE_SIZE), requires_grad=True).to(device)
        onnx_file_name = file_name + "_dynamic.onnx"
        dynamic_axes = {"input": {0: "batch_size"}, "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')

        ONNX = 'trtexec --onnx=%s' % onnx_file_name
        BATCH =  '--explicitBatch'
        ENGINE =  '--saveEngine=%s' % trt_save_name
        WORKSPACE = '--workspace=%d' % space_size
        FP = '--fp16 '
        command = ' '.join([ONNX,BATCH,ENGINE,WORKSPACE,FP])
        os.system(command)

        return onnx_file_name

    else:
        x = torch.randn((batch_size, 3, IN_IMAGE_SIZE, IN_IMAGE_SIZE), requires_grad=True).to(device)
        onnx_file_name = file_name + "_static.onnx"
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')

        ONNX = 'trtexec --onnx=%s' % onnx_file_name
        BATCH = '--explicitBatch'
        ENGINE = '--saveEngine=%s' % trt_save_name
        WORKSPACE = '--workspace=%d' % space_size
        FP = '--fp16 '
        command = ' '.join([ONNX, BATCH, ENGINE, WORKSPACE, FP])
        os.system(command)
        return onnx_file_name



def main(weight_file, cfg_file, batch_size, IN_IMAGE_SIZE, workspace):
    if batch_size <= 0:
        # not developed yet
        onnx_path_demo = transform_to_trt(weight_file, cfg_file, batch_size, IN_IMAGE_SIZE, workspace)
    else:
        # Transform to onnx for demo
        onnx_path_demo = transform_to_trt(weight_file,cfg_file, batch_size, IN_IMAGE_SIZE, workspace)



if __name__ == '__main__':
    print("Converting to onnx and running demo ...")
    args = parse_args()
    main(args.weight, args.model, args.batch, args.img_size, args.workspace)
