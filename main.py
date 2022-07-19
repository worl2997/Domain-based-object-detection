from __future__ import division
from parser import *
from OID_tools.bounding_boxes import *
from train import *
import os
import sys

class PATH():
    def __init__(self):
        self.ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))
        self.DEFAULT_DATA_DIR = os.path.join(self.ROOT_DIR, 'data','custom')
        self.DATA_FILE_DIR = os.path.join(self.ROOT_DIR,'config','custom_data')
        self.cfg_path =  os.path.join(self.ROOT_DIR, 'config','custom_cfg')
        self.model_save_path = os.path.join(self.ROOT_DIR,'weights', 'custom_weight')

if __name__ == '__main__':
    args = parse_arguments()
    path = PATH()
    if args.command == 'downloader':
        domain_groups = bounding_boxes_images(args, path)

    elif args.command == 'train':
        if args.cfg:
            model_cfg = args.cfg
        else:
            # custom된 cfg 파일이 없을경우, 학습하고자 하는 모델의 custom cfg 파일을 만들어줌
            model_cfg = get_custom_cfg(path.cfg_path, args.domain, args.model, args.classes)
        train(args, model_cfg, path)

