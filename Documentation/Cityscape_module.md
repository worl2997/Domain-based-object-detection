# Cityscapes 데이터 지원

본 문서에서는 Segmenation 작업을 위한  Cityscapes 데이터를 전처리하여 object detection 작업을 학습할 수 있는 데이터 셋을 생성하는 코드를 제공한다.  
Reference :  [https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)

## 데이터 다운로드

---

본 프로젝트에서는 Cityscapes 데이터를 전처리하기 용이한 디렉토리 구조로 재구성하여 제공한다. 제공된 cityscape 모듈에서 download_cityscape.py를 실행하면 데이터셋을 다운로드하고 압축해제하여 다음과 같은 파일 구성을 생성한다. 

```bash
cityscape
├── cityscapes
|   ├──gtFine_trainvaltest
|   ├──leftImg8bit_trainvaltest
├── download_cityscape.py
├── data_processing.py
└── Integration_OID.py
```

## 데이터 전처리 및 프레임워크 통합

---

다운로드 된 cityscapes 데이터는 data_processing.py를 통해 segmentation 데이터를 전처리하여 2D bounding box 데이터셋을 생성하고 Integration_OID.py를 통해 가공된 데이터를 프레임워크에 적용할 수 있도록  전처리한다. 

기존의 cityscapes 데이터 셋은 아래 그림과 같은 클래스 그룹을 지원한다. 하지만 명시된 클래스는 segmentation 작업을 위해 지정된 클래스이므로 object detection에는 적합하지 않다. 본 프로젝트에서 제공한 코드는 임의로 크게 4가지 클래스를 지정하여 다음과 같이 구성한다.
![image](/readme/cityscapes_classes.png)

**[ Cityscapes object detection dataset class 목록]**

`['person'**,** 'rider']` ⇒ ‘person’  

`['car'**,** 'truck'**,** 'bus'**,** 'caravan'**,** 'trailer']` ⇒ ‘vehicle’

`['traffic sign']` ⇒  ‘traffic sign’

`['traffic light']` ⇒ ‘traffic light’ 

만약 클래스 목록을 변경하고 싶다면 integration_OID.py에 기재된 class list와  data_processing.py의 convert_annotation 부분을 수정하여야 한다.

```python
# integration_OID.py의 main
if __name__ == '__main__':
    gtFine_dir = "cityscapes/gtFine_trainvaltest/gtFine"
    leftImg_dir = "cityscapes/leftImg8bit_trainvaltest/leftImg8bit"
    # set the save path
    data_path = 'cityscapes_OD_data'
    abs_path = make_abs_path(data_path)
    save_namefile_dir = os.path.join(abs_path,"cityscapes.name")
    data_type = ['test', 'train', 'val']
    # 아래의 클래스 리스트를 수정 
		class_list= ['person','vehicle', 'traffic sign', 'traffic light']

# data_processing.py의 convert_annotation 메소드 
def convert_annotation(json_path,data_path, type):
    os.makedirs(os.path.join(data_path,type,'Label'))
    json_dir = os.path.join(json_path,type)
    file_list = os.listdir(json_dir)
    for filename in file_list:
        load_f = open(os.path.join(json_dir,filename))
        load_dict = json.load(load_f)
        w = load_dict['imgWidth']  # 이미지 넓이 및 높이
        h = load_dict['imgHeight']
        filename = filename[:-21]

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
                cls_id = 1  # 여러 종류의 이동 수단 -> vehicle 카테고리로 묶음
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
```

Cityscapes 데이터는 아래와 같은 명령어를 통해 전처리되고 Object detection 프레임워크에 추가된다. 

```bash
python integration_OID.py 
```

생성된 데이터 구조는 다음과 같으며, 최종 처리된 데이터는 cityscapes_OD_data 디렉토리에 저장된다. trainval_json 폴더는 전처리 과정 중 생긴 디렉토리로 제거해도 무방하다. 

```bash
cityscape
├── cityscapes
│   ├── gtFine_trainvaltest
│   └── leftImg8bit_trainvaltest
├── cityscapes_OD_data
│   ├── cityscapes.name
│   ├── train
│   └── validation
├── data_processing.py
├── download_cityscape.py
├── integration_OID.py
└── trainval_json
    ├── train
    └── validation
```

**[ Cityscapes 데이터 기반 모델 학습 ]**

프레임워크에 통합된 데이터는  cityscape 라는 도메인으로 추가되며, cityscapes 데이터 기반으로 모델 학습을 수행하려면 다음 예시처럼 domain을 cityscapes로 지정해주면 된다. 

```bash
python main.py train --model yolov3 --domain cityscapes --classes 4 --epochs 200 --weights weights/darknet53.conv74 --batch-size 8
```
