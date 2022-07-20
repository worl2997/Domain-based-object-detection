import os
from tqdm import tqdm
from .utils import images_options
from .utils import bcolors as bc
from multiprocessing.dummy import Pool as ThreadPool
import cv2

def make_domain_list(domain_file_path, domain_list):
    # check the 'domain_file_path' whether it is exist or not
    if not os.path.exists(domain_file_path):
        try:
            os.makedirs(domain_file_path)
        except OSError:
            print("Failed to create " + domain_file_path + " directory")
            os.exit(-1)

    # read domains and classes from 'domain_list' file
    group_dict = {}

    for list in domain_list:
        list = list.split(' ')
        list = [x.strip() for x in list] # remove space or newline characters
        list = [x.replace('_', ' ') for x in list if x != ''] # e.g. 'Traffic_light' -> 'Traffic light'

        # make sure the input line contains at lease one class.
        if len(list) < 2:
            break

        domain_name = list[0]
        classes = list[1:]

        group_dict[domain_name] = classes

    # create domain files and write their sub classes on files
    for (domain_name, classes) in group_dict.items():
        with open(os.path.join(domain_file_path, domain_name + '.name'), 'w') as f:
            for c in classes:
                f.write(c + '\n')

    # append the number of classes in front of the class lists
    for domain_name in group_dict.keys():
        class_len = len(group_dict[domain_name])
        group_dict[domain_name].insert(0, class_len)

    print("group_dict:",group_dict)
    # group_dict will be like  -> {'group1' : [2, 'Bus', 'Truck']}, 2 is class number
    return group_dict

def download(args, data_type, df_val, folder, dataset_dir, class_name, class_code, domain_name, domain_dic ,threads = 20):
    
    '''
    Manage the download of the images and the label maker.
    :param args: argument parser.
    :param df_val: DataFrame Values =>  csv 파일의 데이터
    :param folder: train, validation or test
    :param dataset_dir: self explanatory
    :param class_name: self explanatory => Apple, Orange등
    :param class_code: self explanatory => class_dict로 부터 가져온 클라스 코드값
    :param class_list: list of the class if multiclasses is activated => ["Apple", "Orange"]
    :param threads: number of threads
    :return: None
    '''

    if os.name == 'posix':
        rows, columns = os.popen('stty size', 'r').read().split()
    elif os.name == 'nt':
        try:
            columns, rows = os.get_terminal_size(0)
        except OSError:
            columns, rows = os.get_terminal_size(1)
    else:
        columns = 50
    l = int((int(columns) - len(class_name))/2)

    print ('\n' + bc.HEADER + '-'*l + class_name + '-'*l + bc.ENDC)
    print(bc.INFO + 'Downloading {} images.'.format(class_name) + bc.ENDC)
    df_val_images = images_options(df_val, args)
    images_list = df_val_images['ImageID'][df_val_images.LabelName == class_code].values
    images_list = set(images_list)
    print(bc.INFO + '[INFO] Found {} online images for {}.'.format(len(images_list), folder) + bc.ENDC)

    if args.limit is not None:
        import itertools
        if data_type == 'train':
            print(bc.INFO + 'Limiting to {} images.'.format(args.limit) + bc.ENDC)
            images_list = set(itertools.islice(images_list, args.limit))
        else:
            print(bc.INFO + 'Limiting to {} images.'.format(int(args.limit*0.1)) + bc.ENDC)
            images_list = set(itertools.islice(images_list, int(args.limit*0.1)))

    download_img(folder, dataset_dir, domain_name , images_list, threads)
    if not args.sub:
        get_label(folder, dataset_dir, class_name, class_code, df_val, domain_name, domain_dic, args)


def download_img(folder, dataset_dir, domain_name, images_list, threads):
    '''
    Download the images.
    :param folder: train, validation or test
    :param dataset_dir: self explanatory
    :param domain_name: name of domain, ex: highway, Park..
    :param images_list: list of the images to download
    :param threads: number of threads
    :return: None
    '''
    image_dir = folder

    download_dir = os.path.join(dataset_dir, image_dir, domain_name)
    downloaded_images_list = [f.split('.')[0] for f in os.listdir(download_dir)]
    images_list = list(set(images_list) - set(downloaded_images_list))
    pool = ThreadPool(int(threads))

    if len(images_list) > 0:
        print(bc.INFO + 'Download of {} images in {}.'.format(len(images_list), folder) + bc.ENDC)
        commands = []
        for image in images_list:
            path = image_dir + '/' + str(image) + '.jpg ' + '"' + download_dir + '"'
            command = 'aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/' + path                    
            commands.append(command)

        list(tqdm(pool.imap(os.system, commands), total = len(commands) ))

        print(bc.INFO + 'Done!' + bc.ENDC)
        pool.close()
        pool.join()
    else:
        print(bc.INFO + 'All images already downloaded.' +bc.ENDC)


def get_label(folder, dataset_dir, class_name, class_code, df_val, domain_name, domain_dic, args):
    '''
    Make the label.txt files
    :param folder: trai, validation or test
    :param dataset_dir: self explanatory
    :param class_name: self explanatory
    :param class_code: self explanatory
    :param df_val: DataFrame values
    :param class_list: list of the class if multiclasses is activated
    :return: None
    '''
    if not args.noLabels:
        print(bc.INFO + 'Creating labels for {} of {}.'.format(domain_name, folder) + bc.ENDC)

        image_dir = folder #train
        download_dir = os.path.join(dataset_dir, image_dir, domain_name) #custom/train/group1
        label_dir = os.path.join(download_dir, 'Label')

        downloaded_images_list = [f.split('.')[0] for f in os.listdir(download_dir) if f.endswith('.jpg')]
        images_label_list = list(set(downloaded_images_list))
        groups = df_val[(df_val.LabelName == class_code)].groupby(df_val.ImageID)

        classes = domain_dic[domain_name][1:] # 제 1 도메인의 클래스 리스트
        target_class_idx = classes.index(class_name)


        for image in images_label_list:
            try:
                current_image_path = os.path.join(download_dir, image + '.jpg')
                dataset_image = cv2.imread(current_image_path)
                size = dataset_image.shape
                img_height =  1.0 *size[0]
                img_width = 1.0 *size[1]

                boxes = groups.get_group(image.split('.')[0])[['XMin', 'XMax', 'YMin', 'YMax']].values.tolist()
                file_name = str(image.split('.')[0]) + '.txt'
                file_path = os.path.join(label_dir, file_name)


                if os.path.isfile(file_path):
                    f = open(file_path, 'a')
                else:
                    f = open(file_path, 'w')
                # If you want normalize the data to [0,1], do not remove the #
                for box in boxes:
                    # box[0] -> Xmin, box[1] ->Xmax , box[2] -> ymin, box[3] -> ymax

                    # data for darknet
                    abs_x =  box[0] + 0.5*(box[1] - box[0])
                    abs_y =  box[2] + 0.5*(box[3] - box[2])
                    abs_width = box[1] - box[0]
                    abs_height = box[3] - box[2]

                    x = abs_x / img_width
                    y = abs_y / img_height
                    width = abs_width / img_width
                    height = abs_height / img_height

                    # new_box_x = (box[1] + box[0])/2
                    # new_box_y = (box[3] + box[2])/2
                    # new_box_width = box[1] - box[0]
                    # new_box_height = box[3] - box[2]

                    # box[0] *= int(dataset_image.shape[1])
                    # box[1] *= int(dataset_image.shape[1])
                    # box[2] *= int(dataset_image.shape[0])
                    # box[3] *= int(dataset_image.shape[0])
                    # each row in a file is name of the class_name, XMin, YMix, XMax, YMax (left top right bottom)
                    print(target_class_idx, x,y,width,height, file=f)

            except Exception as e:
                pass

        print(bc.INFO + 'Labels creation completed.' + bc.ENDC)
