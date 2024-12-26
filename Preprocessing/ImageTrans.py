import cv2
import json
import os

class ImageTrans:
    def ImageResize(self, img_name, size, data_folder, saved_folder):
        for img in img_name:
            path = data_folder + '/' + img
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            w, h = image.shape[:2]

            # 가로 세로 중 큰 쪽을 기준으로 비율 계산
            scale = size / max(h,w)
            new_width = int(w * scale)
            new_height = int(h * scale)

            ratio = new_width / w
            new_height = int(h * ratio)

            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            saved_path = saved_folder + "/" + img
            cv2.imwrite(saved_path, cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))

class YoloImageTrans:
    def JsonToTxt(self, json_folder, image_folder, image_extension, class_num, saved_folder):
        json_name_list = os.listdir(json_folder)
        for i in json_name_list:
            try:
                #json 파일 열기
                json_path = json_folder + '\\' + i
                with open(json_path, 'r') as file:
                    data = json.load(file)

                # 이미지 파일 열기
                image_path = f'{image_folder}\\{i.replace('.json','')}.{image_extension}'
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
                txt_path = f'{saved_folder}\\{i.replace('.json','')}.txt'
                with open(txt_path, 'w') as f:
                    for line in yolo_data:
                        f.write(f"{line}\n")
            except:
                pass

    
    def ImageResize(self, img_name, size, data_folder, saved_folder):
        for img in img_name:
            path = data_folder + '/' + img
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            w, h = image.shape[:2]

            # 가로 세로 중 큰 쪽을 기준으로 비율 계산
            scale = size / max(h,w)
            new_width = int(w * scale)
            new_height = int(h * scale)

            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # 

            saved_path = saved_folder + "/" + img
            cv2.imwrite(saved_path, cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))