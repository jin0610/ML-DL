### 1-1. YOLOv5 파일 다운로드 
import os
os.getcwd()


### 1-2. yolov5 불러오기 및 필요 패키지 다운로드
## git에서 yolov5 파일 불러오기
# !git clone https://github.com/ultralytics/yolov5.git

## 패키지 다운로드
# !pip install -r ./yolov5/requirements.txt

# 패키지 불러오기
from glob import glob
from sklearn.model_selection import train_test_split 
import cv2
import yaml

### 2-1. 이미지 resize
original_list = glob('./data/original/*.png')
image_list = os.listdir('./data/original/')

for i in range(len(original_list)):
    image = cv2.imread(original_list[i])

    new_size = (640, 640)
    resized_image = cv2.resize(image, new_size)

    path = './data/export/images/' + image_list[i]
    # 덮어쓰기
    cv2.imwrite(path, resized_image)


### 2-2. 학습 데이터셋 분리
img_list = glob('절대경로/*.png')
len(img_list)


## 학습 검정 데이터 분리
val_img_list, test_img_list = train_test_split(img_list, test_size=0.5, random_state=2000)

with open('./data/train.txt', 'w') as f:
    f.write('\n'.join(img_list) + '\n')
    
with open('./data/val.txt', 'w') as f:
    f.write('\n'.join(val_img_list) + '\n')
    
with open('./data/test.txt', 'w') as f:
    f.write('\n'.join(test_img_list) + '\n')

### 2-3. dataset/data.yaml 생성
# 데이터 파일 경로
data = {}
data['train'] = "경로/yolo/data/train.txt"
data['val'] = "경로/yolo/data/val.txt"
data['test'] = "경로/yolo/data/test.txt"

# 라벨
data['names'] = { 0: 'person'}

with open('./data/data.yaml', 'w') as f:
    yaml.dump(data, f)


### 3-1. YOLOv5 모델 학습
# !python ./yolov5/train.py --batch 16 --epoch 50 --data ./data/data.yaml \
# --cfg ./yolov5/models/yolov5s.yaml \
# --weights ./yolov5/yolov5s.pt \
# --name yolov5s_results