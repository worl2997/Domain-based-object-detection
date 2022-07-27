# OID_tools

OID_tools 패키지는 OpenImages Dataset v4 API를 활용하여,
총 600개의 클래스에 대한 1,700,000개 이상의 이미지 및 바운딩 박스 데이터 중 
사용자가 지정하는 클래스의 데이터를 편리하게 다운로드 하는 기능을 제공한다.

- 600개의 클래스중 지정된 클래스에 대한 이미지 다운로드 및 바운딩 박스 생성
- 지정된 여러 클래스를 동시에 다운로드하여 하나의 공통 폴더에 데이터를 생성 
(문서- 프레임워크 셋팅 및 동작에서 domain.txt 관련 예시 참고)
- 모델 학습 모듈에 최적화 된 구조로 데이터 저장 및 데이터 경로가 기재된 파일 생성

OID_tools 패키지를 구성하는 모듈은 다음과 같이 구성되어 있다 

- bounding_boxes.py 
→ domains.txt 파일을 읽어 사용자가 지정한 클래스 데이터를 다운로드하여 도메인 폴더에 저장
- csv_downloader.py
→ 데이터를 다운로드 하는데 필요한 csv 파일이 다운로드 되어있는지를 체크하고 없을시 자동으로 다운로드 지원
- [downloader.py](http://downloader.py) 
→ 이미지 및 라벨 데이터를 다운로드 하는 기능 지원
- [show.py](http://show.py) 
→ 다운로드 과정을 시각화 해주는 기능 지원
- [utils.py](http://utils.py) 
→ 다운로드 폴더 생성 및 이미지 옵션 처리 등 데이터를 다운로드 하는데 필요한 다양한 기능 지원

### **bounding_boxes.py**

---

bounding_boxes.py는 domains.txt 파일을 읽어들여 다운로드할 클래스를 지정하여 OpenImagev4 API로부터 필요한 데이터를 다운로드한다.
아래는 해당 모듈에 대한 코드 및 주석을 나타낸다  

bounding_boxes_images는 다음과 같이 [main.py](http://main.py) (메인 함수)로 부터 각 파일경로를 저장한 클래스와 args 객체를 전달받는다 

```python
def bounding_boxes_images(args,path):
				... 

'''bounding_boxes.py 에서 활용되는 args 인자 (parser.py 참고) '''
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

'''
path 인자 -> 메인함수로 부터 PATH클래스를 전달받음

ROOT_DIR : 프레임워크 프로젝트 디렉토리 
DEFAULT_DATA_DIR : 커스텀 데이터를 관련 파일들을 다운로드 할 디렉토리 
DATA_FILE_DIR : 모델 학습 모듈이 읽어들일 data 파일을 저장할 경로 
cfg_path : custom model에 대한 cfg 파일을 저장할 경로 
model_save_path : 학습된 커스텀 모델 가중치 파일이 저장될 경로 
'''
class PATH():
    def __init__(self):
        self.ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))
        self.DEFAULT_DATA_DIR = os.path.join(self.ROOT_DIR, 'data','custom')
        self.DATA_FILE_DIR = os.path.join(self.ROOT_DIR,'config','custom_data')
        self.cfg_path =  os.path.join(self.ROOT_DIR, 'config','custom_cfg')
	        self.model_save_path = os.path.join(self.ROOT_DIR,'weights', 'custom_weight')

```

인자로 전달 받은 args.parser와 PATH 클래스를 기반으로 다음과 같은 코드를 수행한다.

```python
def bounding_boxes_images(args,path):
    root_dir = path.ROOT_DIR
    default_oid_dir = path.DEFAULT_DATA_DIR

		# 데이터 및 csv 파일 저장 경로 지정  
    if not args.OID_CSV:
        dataset_dir = default_oid_dir  # ../data/custom
        csv_dir = os.path.join(default_oid_dir, 'csv_folder')
    else:
        dataset_dir = os.path.join(default_oid_dir, args.Dataset)
        csv_dir = os.path.join(default_oid_dir, 'csv_folder')
		
		# OpenImagev4 API로부터 다운로드 가능한 클래스 이름과 코드를 기록해 놓은 csv 파일
    name_file_class = 'class-descriptions-boxable.csv'
    classes_csv_path = os.path.join(csv_dir, name_file_class)

    logo(args.command)

		# train 및 validation 데이터 관련 정보를 기록해 놓은 csv 파일  
    folder = ['train', 'validation']
    file_list = ['train-annotations-bbox.csv', 'validation-annotations-bbox.csv']

		# domains.txt를 읽어들여 다운로드 할 클래스 리스트 생성 
    if args.dm_list[0].endswith('.txt'):
        with open(args.dm_list[0]) as f:
            args.dm_list = f.readlines()
            args.dm_list = [x.strip() for x in args.dm_list]
        print('download classes: ' + str(args.dm_list))

    domain_list = args.dm_list
    name_file_path = os.path.join(default_oid_dir, 'domain_list')

    print("name_file_path:", name_file_path)
    print("domain_list:", domain_list)
		
		# domain_dict 생성 예시->  {'domain_1 ' : [2, 'Bus', 'Truck']}  (2 는 클래스 개수)
    domain_dict = make_domain_list(name_file_path, domain_list)
		# csv파일, 이미지, 라벨 데이터를 저장할 디렉토리 생성 (util.py 참조)  
    mkdirs(dataset_dir, csv_dir, domain_dict)  
		# class-descriptions-boxable.csv 파일 존재 여부 확인 및 다운로드 
    error_csv(name_file_class, csv_dir, args.yes)

		# train-annotations-bbox.csv, validation-annotations-bbox.csv 파일 존재여부 확인 및 다운ㄹ 
		# 각 파일 경로를 csv_file_list에 저장 
    csv_file_list = []
    for i in range(2):
       name_file = file_list[i]
       csv_file = TTV(csv_dir, name_file, args.yes)
       csv_file_list.append(csv_file)

    for domain_name, class_list in domain_dict.items():
        print(bcolors.INFO + 'Downloading {} together.'.format(str(class_list[1:])) + bcolors.ENDC)
        df_classes = pd.read_csv(classes_csv_path, header=None)
        class_dict = {}

				# 다운로드 할 클래스 이름과 코드를 딕셔너리로 저장 
        # class_dict => : {'Orange': '/m/0cyhj_', 'Apple': '/m/014j1m'}
        for class_name in class_list[1:]:
            class_dict[class_name] = df_classes.loc[df_classes[1] == class_name].values[0][0]

				# train 및 valid csv 파일을 참조하여 사용자가 지정한 클래스의 데이터를 다운로드 
        for class_name in class_list[1:]:
            for i in range(2):
                name_file = csv_file_list[i].split('/')[-1]
                df_val = pd.read_csv(csv_file_list[i])
                data_type = name_file[:5]  # train or valid

                if not args.n_threads: # 데이터 다운로드에 쓰일 thread 지정 (default: 100) 
                    download(args, data_type, df_val, folder[i], dataset_dir, class_name, class_dict[class_name],
                             domain_name, domain_dict)
                else:

                    download(args, data_type, df_val, folder[i], dataset_dir, class_name, class_dict[class_name],
                             domain_name, domain_dict, args.n_threads)
				
				# 학습에 필요한 파일들 경로가 기재되어 있는 data 파일을 생성 
        make_data_file(root_dir, default_oid_dir, domain_dict)
	
    return domain_dict
```

## csv_download.py

---

csv_download.py 모듈은 데이터를 다운로드 하는데 필요한 csv파일의 존재 여부를 확인하고, 다운로드하는 기능을 지원한다. 필요한 csv 파일 목록은 아래와 같다 .

- class-descriptions-boxable.csv
- train-annotations-bbox.csv
- validation-annotations-bbox.csv

### class-descriptions-boxable.csv 예시

- 다운로드 가능한 detection 데이터 클래스들을 기재한 파일로 해당 class의 코드 (예 : Apple → /m/014j1m) 를 통해 API에 접근하여 데이터를 다운로드한다

```
/m/011k07,Tortoise
/m/011q46kg,Container
/m/012074,Magpie
/m/0120dh,Sea turtle
/m/01226z,Football
/m/012n7d,Ambulance
/m/012w5l,Ladder
/m/012xff,Toothbrush
/m/012ysf,Syringe
/m/0130jx,Sink
/m/0138tl,Toy
/m/013y1f,Organ
/m/01432t,Cassette deck
/m/014j1m,Apple
...
```

### train/valid-annotations-bbox.csv 예시

- OpenImage 데이터 셋에 존재하는 데이터 정보를 저장한 csv파일
- 이미지 ID와 해당 이미지에 존재하는 바운딩 박스 정보, 박스가 나타내는 클래스 정보 및 데이터 옵션에 대한 정보가 기재되어있다
- 데이터 옵션 정보
    - 아래 옵션들은 command 입력에서 0 또는 1값을 지정함으로써 사용자가 필요에 맞는 이미지 데이터를 선택하여 다운로드 할 수 있다 (parser.py 참고)
    - **IsOccluded**
        
        : 객체가 이미지의 다른 객체에 의해 가려져 있음을 나타냄
        
    - **IsTruncated**
        
        : 객체가 이미지의 경계를 넘어 확장됨을 의미
        
    - **IsGroupOf**
        
        : 박스가 여러 객체 그룹에 걸쳐 있음을 나타냄, 즉 여러 객체가 서로 밀접하게 접촉되어있어 심하게 가리는 데이터임을 나타낸다. 
        
    - **IsDepiction**
        
        : 실제 물리적 인스턴스가 아닌 만화 또는 그림으로 표현된 객체 이미지
        
    - **IsInside**
        
        : 물체의 내부(예: 자동차 내부 또는 건물 내부)에서 촬영한 이미지를 나타냄
        

```
ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
000002b66c9c498e,xclick,/m/01g317,1,0.012500,0.195312,0.148438,0.587500,0,1,0,0,0
000002b66c9c498e,xclick,/m/01g317,1,0.025000,0.276563,0.714063,0.948438,0,1,0,0,0
000002b66c9c498e,xclick,/m/01g317,1,0.151562,0.310937,0.198437,0.590625,1,0,0,0,0
000002b66c9c498e,xclick,/m/01g317,1,0.256250,0.429688,0.651563,0.925000,1,0,0,0,0
000002b66c9c498e,xclick,/m/01g317,1,0.257812,0.346875,0.235938,0.385938,1,0,0,0,0
000002b66c9c498e,xclick,/m/01g317,1,0.320312,0.368750,0.260938,0.328125,1,0,0,0,0
000002b66c9c498e,xclick,/m/01g317,1,0.412500,0.945312,0.120312,0.475000,1,0,1,0,0
000002b66c9c498e,xclick,/m/0284d,1,0.528125,0.923437,0.675000,0.964063,0,0,0,0,0
...
```