import cv2

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
