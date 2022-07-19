from utils import *
from .downloader import *
from .csv_downloader import *
from .utils import *
import os


def bounding_boxes_images(args,path):
    root_dir = path.ROOT_DIR
    default_oid_dir = path.DEFAULT_DATA_DIR
    if not args.OID_CSV:
        dataset_dir = default_oid_dir  # ../data/custom
        csv_dir = os.path.join(default_oid_dir, 'csv_folder')
    else:
        dataset_dir = os.path.join(default_oid_dir, args.Dataset)
        csv_dir = os.path.join(default_oid_dir, 'csv_folder')

    name_file_class = 'class-descriptions-boxable.csv'
    classes_csv_path = os.path.join(csv_dir, name_file_class)

    logo(args.command)
    folder = ['train', 'validation']
    file_list = ['train-annotations-bbox.csv', 'validation-annotations-bbox.csv']

    if args.dm_list[0].endswith('.txt'):
        with open(args.dm_list[0]) as f:
            args.dm_list = f.readlines()
            args.dm_list = [x.strip() for x in args.dm_list]
        print('download classes: ' + str(args.dm_list))

    domain_list = args.dm_list
    name_file_path = os.path.join(default_oid_dir, 'domain_list')
    # domain_list =>  ['group1 Orange Apple', 'group2 Bus Traffic_light Car Fire_hydrant']

    print("name_file_path:", name_file_path)
    print("domain_list:", domain_list)
    domain_dict = make_domain_list(name_file_path, domain_list)
    mkdirs(dataset_dir, csv_dir, domain_dict)
    error_csv(name_file_class, csv_dir, args.yes)

    csv_file_list = []
    for i in range(2):
       name_file = file_list[i]
       csv_file = TTV(csv_dir, name_file, args.yes)
       csv_file_list.append(csv_file)

    # create class list file for each domain in data/custom/domain_list directory

    for domain_name, class_list in domain_dict.items():
        print(bcolors.INFO + 'Downloading {} together.'.format(str(class_list[1:])) + bcolors.ENDC)
        df_classes = pd.read_csv(classes_csv_path, header=None)
        class_dict = {}
        # class_dict => : {'Orange': '/m/0cyhj_', 'Apple': '/m/014j1m'}

        for class_name in class_list[1:]:
            class_dict[class_name] = df_classes.loc[df_classes[1] == class_name].values[0][0]

        for class_name in class_list[1:]:
            for i in range(2):
                name_file = csv_file_list[i].split('/')[-1]
                df_val = pd.read_csv(csv_file_list[i])
                data_type = name_file[:5]  # train or valid

                if not args.n_threads:
                    download(args, data_type, df_val, folder[i], dataset_dir, class_name, class_dict[class_name],
                             domain_name, domain_dict)
                else:

                    download(args, data_type, df_val, folder[i], dataset_dir, class_name, class_dict[class_name],
                             domain_name, domain_dict, args.n_threads)

        make_data_file(root_dir, default_oid_dir, domain_dict)

    return domain_dict
