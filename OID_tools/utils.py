import os
'''데이터셋을 파싱하거나 다운로드 하는데 필요한 유틸 함수들 '''


def make_img_list_txt(data_dir, domain_name, train):
    if train:
        path_ = os.path.join(data_dir, 'train', domain_name)
        file_name = domain_name + '_train.txt'
    else:
        path_ = os.path.join(data_dir, 'validation', domain_name)
        file_name = domain_name + '_valid.txt'
    img_list = os.listdir(path_)
    img_list.remove('Label')
    file_path_list = [os.path.join(path_, x) +'\n' for x in img_list if x != file_name]

    f = open(os.path.join(path_, file_name), 'w')
    f.writelines(file_path_list)
    f.close()
    return os.path.join(path_, file_name)


def make_data_file(root_dir, default_oid_dir, domain_dict):
    print('in make config, domain_dict:' + str(domain_dict))
    config_file_path = os.path.join(root_dir, 'config', 'custom_data')

    if not os.path.isdir(config_file_path):
        os.makedirs(config_file_path)

    for domain_name, class_list in domain_dict.items():
        num_classes = class_list[0]

        train_txt_path = make_img_list_txt(default_oid_dir, domain_name, True)
        valid_txt_path = make_img_list_txt(default_oid_dir, domain_name, False)

        n_file_name = '%s.name' % domain_name
        names = os.path.join(default_oid_dir, 'domain_list', n_file_name)
        data_file_name = "%s.data" % domain_name

        with open(os.path.join(config_file_path, data_file_name), 'w') as f:
            f.write('classes=' + str(num_classes) + '\n')
            f.write('train=' + train_txt_path + '\n')
            f.write('valid=' + valid_txt_path + '\n')
            f.write('names=' + names + '\n')
            f.write('backup= backup/' + '\n')
            f.write('eval=coco' + '\n')



def get_domain_group(DEFAULT_DATA_DIR, domain):
    list_path = os.path.join(DEFAULT_DATA_DIR, domain)
    domain_dict = {}
    n_file = os.listdir(list_path)
    print(n_file)
    fp = open(os.path.join(list_path, n_file), "r")
    names = fp.read().split("\n")  # 뭐 class 이름적힌 리스트 만들어줌
    domain_dict[n_file[:-5]] = [len(names) - 1] + names[:-1]
    return domain_dict


def parse_custom_data(custom_path, group_name):
    train_path = os.path.join(custom_path, 'train', group_name)
    valid_path = os.path.join(custom_path, 'validation', group_name)
    name_path = os.path.join(custom_path, 'domain_list', '%s.name' % group_name)

    # train_flie_list = os.listdir(train_path).remove('Label')
    # valid_flie_list = os.listdir(valid_path).remove('Label')
    return train_path, valid_path, name_path


# 훈련시키려는 도메인의 클래스 개수에 따라 yolo-tiny,
# 혹은 yolov3의 cfg파일을 베이스로 새 model-cfg파일을 생성하고 경로반환함
def get_custom_cfg(path, domain, model_type, class_num):
    if model_type == "yolov3_tiny":
        custom_path = os.path.join(path, 'create_custom_yolov3_tiny.sh')
        cfg_name = domain +'_yolov3_tiny_' + str(class_num) + '.cfg'
        if cfg_name in os.listdir(path):
            return os.path.join(path, cfg_name)
        else:
            os.system("bash %s %d %s %s" % (custom_path, int(class_num), model_type, domain))
            os.system("mv %s %s" % (cfg_name, path))
            return os.path.join(path, cfg_name)
    elif model_type == "yolov3":
        custom_path = os.path.join(path, 'create_custom_yolov3.sh')
        cfg_name =  domain +'_yolov3_' + str(class_num) + '.cfg'
        if cfg_name in os.listdir(path):
            return os.path.join(path, cfg_name)
        else:
            os.system("bash %s %d %s %s" % (custom_path, int(class_num),model_type, domain ))
            os.system("mv %s %s" % (cfg_name, path))
            return os.path.join(path, cfg_name)
    elif model_type == "lw_yolo":
        custom_path = os.path.join(path, 'create_custom_lw_yolo.sh')
        cfg_name = domain + '_lw_yolo_' + str(class_num) + '.cfg'
        if cfg_name in os.listdir(path):
            return os.path.join(path, cfg_name)
        else:
            os.system("bash %s %d %s %s" % (custom_path, int(class_num), model_type, domain))
            os.system("mv %s %s" % (cfg_name, path))
            return os.path.join(path, cfg_name)

#
def images_options(df_val, args):
    '''
    Manage the options for the images downloader.
    :param df_val: DataFrame Value.
    :param args: argument parser.
    :return: modified df_val
    '''
    if args.image_IsOccluded is not None:
        rejectedID = df_val.ImageID[df_val.IsOccluded != int(args.image_IsOccluded)].values
        df_val = df_val[~df_val.ImageID.isin(rejectedID)]

    if args.image_IsTruncated is not None:
        rejectedID = df_val.ImageID[df_val.IsTruncated != int(args.image_IsTruncated)].values
        df_val = df_val[~df_val.ImageID.isin(rejectedID)]

    if args.image_IsGroupOf is not None:
        rejectedID = df_val.ImageID[df_val.IsGroupOf != int(args.image_IsGroupOf)].values
        df_val = df_val[~df_val.ImageID.isin(rejectedID)]

    if args.image_IsDepiction is not None:
        rejectedID = df_val.ImageID[df_val.IsDepiction != int(args.image_IsDepiction)].values
        df_val = df_val[~df_val.ImageID.isin(rejectedID)]

    if args.image_IsInside is not None:
        rejectedID = df_val.ImageID[df_val.IsInside != int(args.image_IsInside)].values
        df_val = df_val[~df_val.ImageID.isin(rejectedID)]

    return df_val


# Dataset_folder -> data/custom
def mkdirs(Dataset_folder, csv_folder, domain_dict):
    directory_list = ['train', 'validation']

    for directory in directory_list:
        for group_name in domain_dict:
            if not Dataset_folder.endswith('_nl'):
                folder = os.path.join(Dataset_folder, directory, group_name, 'Label')
            else:
                folder = os.path.join(Dataset_folder, directory, group_name, 'Label')
            if not os.path.exists(folder):
                os.makedirs(folder)
            filelist = [f for f in os.listdir(folder) if f.endswith(".txt")]
            for f in filelist:
                os.remove(os.path.join(folder, f))

    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)


# 다운로드 진행상항 출력
def progression_bar(total_images, index):
    # 윈도우에서
    if os.name == 'nt':
        from ctypes import windll, create_string_buffer

        h = windll.kernel32.GetStdHandle(-12)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)

        if res:
            import struct
            (bufx, bufy, curx, cury, wattr,
             left, top, right, bottom, maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
            columns = right - left + 1
            rows = bottom - top + 1
        else:
            columns, rows = 80, 25  # can't determine actual size - return default values
    # 리눅스에서
    else:
        rows, columns = os.popen('stty size', 'r').read().split()
    toolbar_width = int(columns) - 10
    image_index = index
    index = int(index / total_images * toolbar_width)

    print(' ' * (toolbar_width), end='\r')
    bar = "[{}{}] {}/{}".format('-' * index, ' ' * (toolbar_width - index), image_index, total_images)
    print(bar.rjust(int(columns)), end='\r')


def show_classes(classes):
    for n in classes:
        print("- {}".format(n))
    print("\n")


def logo(command):
    bc = bcolors

    print(bc.OKGREEN + """
		   ___      ___   _____  __       
		 .'   `.  .'   `.|_   _| \ \     [  ]
		/  .-.__\/  .-.__\ | |    \ \    / /
		| |   _ _| |   _ _ | |     \ \  / /
		\  `-'  /\  `-'  /_| |_     \ \/ /  
		 `.___.'  `.___.'|_____|     \__/  
	""" + bc.ENDC)

    if command == 'downloader':
        print(bc.OKGREEN + '''
             _____                    _                 _             
            (____ \                  | |               | |            
            | |   \ \ ___  _ _ _ ____ | | ___   ____  _ | | ____  ____ 
            | |   | / _ \| | | |  _ \| |/ _ \ / _  |/ || |/ _  )/ ___)
            | |__/ / |_| | | | | | | | | |_| ( ( | ( (_| ( (/ /| |    
            |_____/ \___/ \____|_| |_|_|\___/ \_||_|\____|\____)_|    

        ''' + bc.ENDC)


class bcolors:
    HEADER = '\033[95m'

    INFO = '    [INFO] | '
    OKBLUE = '\033[94m[DOWNLOAD] | '
    WARNING = '\033[93m    [WARN] | '
    FAIL = '\033[91m   [ERROR] | '

    OKGREEN = '\033[92m'
    ENDC = '\033[0m'