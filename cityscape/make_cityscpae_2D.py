import json
import os
import shutil
import os.path
from json2txt import make_json2txt

def gather_json(gtFine_dir,type):
    if type == 'train' or type =='test':
        dest_dir = 'trainval_json/train'
    else:
        dest_dir = 'trainval_json/validation'
    dir = os.path.join(gtFine_dir, type)
    regions = os.listdir(dir)

    if os.path.isdir(dest_dir):
        pass
    else:
        os.makedirs(dest_dir)


    for region in regions:
        json_dir = os.path.join(dir,region)
        file_list = [x for x in os.listdir(json_dir) if x[-4:]=='json']
        for i in file_list:
            shutil.copy(os.path.join(json_dir,i), os.path.join(dest_dir,i))

def gather_png(leftImg_dir, data_path,type):
    if type == 'train' or type =='test':
        dest_dir = os.path.join(data_path,'train')
    else:
        dest_dir = os.path.join(data_path,'validation')

    dir = os.path.join(leftImg_dir, type)
    regions = os.listdir(dir)
    if os.path.isdir(dest_dir):
        pass
    else:
       os.makedirs(dest_dir)

    for region in regions:
        img_dir = os.path.join(dir,region)
        file_list = os.listdir(img_dir)
        for i in file_list:
            shutil.copy(os.path.join(img_dir,i), os.path.join(dest_dir,i))

def make_namefile(save_dir):
    namefile = open(save_dir, 'w')  # left8bit 이미지에 대한 라벨 생성
    class_list= ['person','vehicle', 'traffic sign', 'traffic light']
    for class_ in class_list:
        namefile.write(class_ + '\n')
    return


def make_datafile(abs_dir,train_dir, val_dir,namefile_dir, num_of_class):
    a = abs_dir.split('/')
    new_dir = a[:-2]
    proj_dir = '/'.join(new_dir)
    save_dir = os.path.join(proj_dir,'config','custom_data','cityscapes.data')

    with open(save_dir,'w') as f:
        f.write("classes=%d"%(num_of_class)+'\n')
        f.write("train=%s"%(train_dir)+'\n')
        f.write("valid=%s" % (val_dir) + '\n')
        f.write("names=%s" % (namefile_dir) + '\n')
        f.write("backup=backup/"+'\n')
        f.write("eval=coco" + '\n')


def make_abs_path(data_path):
    a = os.path.abspath(__file__)
    b = a.split('/')
    b.pop(-1)
    abs_path = os.path.join('/'.join(b), data_path)
    return abs_path

def make_filelist_txt(data_path, abs_path ):
    traintxt_dir = os.path.join(abs_path,'train')
    valtxt_dir = os.path.join(abs_path,'validation')

    t_list = os.listdir(traintxt_dir)
    v_list = os.listdir(valtxt_dir)

    train_txt_path = os.path.join(traintxt_dir,'cityscapes_train.txt')
    val_txt_path = os.path.join(valtxt_dir,'cityscapes_validation.txt')

    with open(train_txt_path, 'w') as ft:
        for i in t_list:
            ft.write(os.path.join(abs_path,'train',i)+'\n')
    with open(val_txt_path, 'w') as fv:
        for i in v_list:
            fv.write(os.path.join(abs_path,'validation',i)+'\n')

    return  train_txt_path, val_txt_path


if __name__ == '__main__':
    gtFine_dir = "cityscapes/gtFine_trainvaltest/gtFine"
    leftImg_dir = "cityscapes/leftImg8bit_trainvaltest/leftImg8bit"
    # set the save path
    data_path = 'cityscapes_OD_data'
    abs_path = make_abs_path(data_path)
    save_namefile_dir = os.path.join(abs_path,"cityscapes.name")

    num_of_class= 4
    data_type = ['test', 'train', 'val']

    for type in data_type:
        gather_json(gtFine_dir, type)
        gather_png(leftImg_dir,data_path, type)

    train_txt, val_txt = make_filelist_txt(data_path, abs_path)
    make_json2txt(data_path)
    make_namefile(save_namefile_dir)
    make_datafile(abs_path, train_txt, val_txt, save_namefile_dir,num_of_class)