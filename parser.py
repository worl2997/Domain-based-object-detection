import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Open Image Dataset Downloader  & train')
    parser.add_argument("command",
                        metavar="<command> 'downloader', 'train'",
                        help="'downloader' or 'train'")

    ####################################
    # settings for training
    ####################################
    parser.add_argument("--model", type=str, default=None,
                       help="which model you want to make cfg file ex: yolov3, yolov3_tiny, lw_yolo")
    parser.add_argument("--domain", type=str, default=None, help="domain name for train")
    parser.add_argument("--custom", type=bool, default=True, help="custom train or not")
    parser.add_argument("--classes", type=int, default=80, help="Number of classes for training")
    parser.add_argument("-cfg", "--cfg", type=str, default=None, help="Path to model definition file (.cfg)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=2, help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=2, help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.3, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")


    ####################################
    # OpenImage dataset download settings
    ####################################
    parser.add_argument('--limit', required=False, type=int, default=None,
                        metavar="integer number",
                        help='Optional limit on number of images to download')
    parser.add_argument('--dm_list', required=False, default='domains.txt',nargs='+',
                        metavar="list of classes",
                        help="Sequence of 'strings' of the wanted classes")
    parser.add_argument('--noLabels', required=False, action='store_true',
                        help='No labels creations')
    parser.add_argument('--n_threads', required=False, metavar="[default 20]", default=100,
                        help='Num of the threads for download dataset')


    # Not essential options below
    parser.add_argument('-y', '--yes', required=False, action='store_true',
                        # metavar="Yes to download missing files",
                        help='ans Yes to possible download of missing files')
    parser.add_argument('--sub', required=False, choices=['h', 'm'],
                        metavar="Subset of human verified images or machine generated (h or m)",
                        help='Download from the human verified dataset or from the machine generated one.')
    parser.add_argument('--image_IsOccluded', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates that the object is occluded by another object in the image.')
    parser.add_argument('--image_IsTruncated', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates that the object extends beyond the boundary of the image.')
    parser.add_argument('--image_IsGroupOf', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates that the box spans a group of objects (min 5).')
    parser.add_argument('--image_IsDepiction', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates that the object is a depiction.')
    parser.add_argument('--image_IsInside', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates a picture taken from the inside of the object.')
    parser.add_argument('--OID_CSV', required=False,
                        metavar="/path/to/custom/csv/",
                        help='Directory of the OID dataset folder')
    return parser.parse_args()



