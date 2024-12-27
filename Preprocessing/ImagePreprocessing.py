import cv2
import numpy as np
from glob import glob
import os

class ImageTrans:
    def ImageResize(self, image_folder, size, saved_folder):
        image_name_list = os.listdir(image_folder)
        for image_name in image_name_list:
            image_path = image_folder + '/' + image_name
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    def ImagePadding(self, image_folder, size, saved_folder):
        image_name_list = os.listdir(image_folder)
        for image_name in image_name_list:
            image_path = image_folder + '/' + image_name
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    def ImageRotate(self, image_folder, size, saved_folder):
        image_name_list = os.listdir(image_folder)
        for image_name in image_name_list:
            image_path = image_folder + '/' + image_name
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width = image.shape[:2]

            # 가로 세로 중 큰 쪽을 기준으로 비율 계산
            scale = size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)

            # 이미지 비율을 유지하며 크기 조정
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # 이미지의 중심을 중심으로 이미지를 45도 회전합니다.
            M = cv2.getRotationMatrix2D((new_width//2, new_height//2), 45, 0.4)
            rotate_img = cv2.warpAffine(resized_image, M, (new_width, new_height))

            # 이미지 저장
            new_path = saved_folder + '/' + image_name
            cv2.imwrite(new_path, rotate_img)





