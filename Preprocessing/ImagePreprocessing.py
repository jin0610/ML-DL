import cv2
import numpy as np
from glob import glob
import json
import os

def FileExtensions(image_folder):
    file_path = image_folder + '/*'
    files = glob(file_path)
    for name in files:
        if not os.path.isdir(name):
            src = os.path.splitext(name)
            if src[1] in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png','.PNG']:
                os.rename(name, src[0] + '.png')


def ImageResize(image_folder, size, saved_folder):
    image_name_list = os.listdir(image_folder)
    for image_name in image_name_list:
        image_path = image_folder + '/' + image_name
        image = cv2.imread(image_path)

        height, width = image.shape[:2]

        # 가로 세로 중 큰 쪽을 기준으로 비율 계산
        scale = size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)

        # 이미지 비율을 유지하며 크기 조정
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # 이미지 저장
        new_path = saved_folder + '/' + image_name
        cv2.imwrite(new_path, resized_image)
    print("Resize Image Success!")

# 이미지 패딩
def ImagePadding(image_folder, size, saved_folder):
    image_name_list = os.listdir(image_folder)
    for image_name in image_name_list:
        image_path = image_folder + '/' + image_name
        image = cv2.imread(image_path)

        height, width = image.shape[:2]

        # 가로 세로 중 큰 쪽을 기준으로 비율 계산
        scale = size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)

        # 이미지 비율을 유지하며 크기 조정
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # 중앙에 이미지를 배치하고 패딩 추가
        top_pad = (size - new_height) // 2
        bottom_pad = size - new_height - top_pad
        left_pad = (size - new_width) // 2
        right_pad = size - new_width - left_pad

        padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0,0,0])
        # 이미지 저장
        new_path = saved_folder + '/' + image_name
        cv2.imwrite(new_path, padded_image)
    print("Padding Success!")

# 이미지 회전전
def ImageRotate(image_folder, size, angle, scale, saved_folder):
    image_name_list = os.listdir(image_folder)
    for image_name in image_name_list:
        image_path = image_folder + '/' + image_name
        image = cv2.imread(image_path)

        height, width = image.shape[:2]

        # 가로 세로 중 큰 쪽을 기준으로 비율 계산
        scale = size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)

        # 이미지 비율을 유지하며 크기 조정
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # 이미지의 중심을 중심으로 이미지를 45도 회전합니다.
        # cv2.getRotationMatrix2D(center, angle, scale) -> center : 회전 중심좌표(x,y), 
        # angle : (반시계 방향) 회전 각도 / 음수는 시계방향 ex) 45, -45
        # scale : 추가적인 확대 비율 ex) 0.4
        M = cv2.getRotationMatrix2D((new_width//2, new_height//2), angle, scale)
        rotate_img = cv2.warpAffine(resized_image, M, (new_width, new_height))

        # 이미지 저장
        new_image_name = "{0}_{1}.{2}".format(image_name.split('.')[0], angle, image_name.split('.')[-1])
        new_path = saved_folder + '/' + new_image_name
        print(new_image_name)
        cv2.imwrite(new_path, rotate_img)
    print("Image Rotation Success!")

# json 파일마다 수정 필요
def JsonToTxt(self, json_folder, image_folder, image_extension, class_num, saved_folder):
    json_name_list = os.listdir(json_folder)
    for i in json_name_list:
        try:
            #json 파일 열기
            json_path = json_folder + '\\' + i
            with open(json_path, 'r') as file:
                data = json.load(file)

            # 이미지 파일 열기
            image_path = "{0}\\{1}.{2}".format(image_folder, i.replace('.json', ''),{image_extension})
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            yolo_data = []

            for box in data['annotations']['bbox']:
                x, y, w, h = int(box['x']), int(box['y']), int(box['w']), int(box['h'])

                # 중심 좌표와 너비, 높이를 정규화
                x_center = (x + w / 2) / w
                y_center = (y + h / 2) / h

                yolo_data.append(f"{class_num} {x_center} {y_center} {w} {h}")

            # YOLO 데이터를 txt 파일로 저장
            txt_path = "{0}\\{1}.txt".format(saved_folder, i.replace('.json',''))
            with open(txt_path, 'w') as f:
                for line in yolo_data:
                    f.write(f"{line}\n")
        except:
            pass



