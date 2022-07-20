import json
import os
import shutil

def position(pos):
    # poygon 라벨로 부터 bbox의 xmin, ymin, xmax, ymax를 찾는 함수
    x = []
    y = []
    nums = len(pos)
    for i in range(nums):
        x.append(pos[i][0])
        y.append(pos[i][1])
    x_max = max(x)
    x_min = min(x)
    y_max = max(y)
    y_min = min(y)
    # print(x_max,y_max,x_min,y_min)
    b = (float(x_min), float(y_min), float(x_max), float(y_max))
    # print(b)
    return b

def convert(size, box):
    # xmin,ymin,xmax,ymax 로 구성됨
    # size[0] -> width
    # size[1] -> height

    image_width = 1.0 * size[0]
    image_height = 1.0 * size[1]

    absolute_x = box[0] + 0.5 * (box[2] - box[0])
    absolute_y = box[1] + 0.5 * (box[3] - box[1])

    absolute_width = box[2] - box[0]
    absolute_height = box[3] - box[1]

    x = absolute_x / image_width
    y = absolute_y / image_height
    width = absolute_width / image_width
    height = absolute_height / image_height

    return (x, y, width, height)


def convert_annotation(json_path,data_path, type):
    os.makedirs(os.path.join(data_path,type,'Label'))
    json_dir = os.path.join(json_path,type)
    file_list = os.listdir(json_dir)
    for filename in file_list:
        load_f = open(os.path.join(json_dir,filename))
        load_dict = json.load(load_f)
        w = load_dict['imgWidth']  # 이미지 넓이 및 높이
        h = load_dict['imgHeight']
        filename = filename[:-16]

        # set the save path for converted annotation file
        out_file = open('%s/%s/Label/%s_leftImg8bit.txt' % (data_path,type,filename), 'w')
        objects = load_dict['objects']
        nums = len(objects)  #  object 수
        cls_id = ''
        for i in range(0, nums):
            labels = objects[i]['label']
            # print(i)
            if (labels in ['person', 'rider']):
                # print(labels)
                pos = objects[i]['polygon']
                bb = position(pos)
                bb = convert((w, h), bb)
                cls_id = 0  # 보행자 및 탑승자들을 person으로 묶음
                out_file.write("%d %.6f %.6f %.6f %.6f" % (cls_id, bb[0], bb[1], bb[2], bb[3]) +'\n' )
                # print(type(pos))
            elif (labels in ['car', 'truck', 'bus', 'caravan', 'trailer']):
                # print(labels)
                pos = objects[i]['polygon']
                bb = position(pos)
                bb = convert((w, h), bb)
                cls_id = 1  # 여러 종류의 탈 -> vehicle 카테고리로 묶어버림
                out_file.write("%d %.6f %.6f %.6f %.6f" % (cls_id, bb[0], bb[1], bb[2], bb[3]) +'\n' )
            elif (labels in ['traffic sign']):
                # print(labels)
                pos = objects[i]['polygon']
                bb = position(pos)
                bb = convert((w, h), bb)
                cls_id = 2
                out_file.write("%d %.6f %.6f %.6f %.6f" % (cls_id, bb[0], bb[1], bb[2], bb[3]) +'\n' )
            elif (labels in ['traffic light']):
                # print(labels)
                pos = objects[i]['polygon']
                bb = position(pos)
                bb = convert((w, h), bb)
                cls_id = 3
                out_file.write("%d %.6f %.6f %.6f %.6f" % (cls_id, bb[0], bb[1], bb[2], bb[3]) +'\n' )

        if cls_id == '':
            print('no label json:', "%s_gtFine_polygons.json" % (filename))



def make_json2txt(data_path):
    dtype = ['train','validation']
    json_path = 'trainval_json'
    for type in dtype:
        convert_annotation(json_path, data_path, type)




