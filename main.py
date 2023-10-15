import os
import cv2
from align_trans import Face_alignment
import torch
from PIL import Image, ImageDraw, ImageFont
from MTCNN import create_mtcnn_net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parent_directory ="C:\\Users\Admin\Downloads\input"
op = "C:\\Users\Admin\Downloads\\output"
import numpy as np
def detec_and_align(img_path):
        img = cv2.imread(img_path)
        try:
            check =img[:]
            if img.shape != None:
                bboxes, landmarks = create_mtcnn_net(check, 20, device,
                                                     p_model_path='Weights/pnet_Weights',
                                                     r_model_path='Weights/rnet_Weights',
                                                     o_model_path='Weights/onet_Weights')
                check = Face_alignment(check, default_square=True, landmarks=landmarks)
                no_align = img[bboxes[0][2]:bboxes[0][0],bboxes[0][1]:bboxes[0][3]]
                return check[0]


                # print("chưa cắt")
        except Exception as e :
            try:
                check = img[50:350,50:350]
                if check.shape != None:
                    bboxes, landmarks = create_mtcnn_net(check, 20, device,
                                                         p_model_path='Weights/pnet_Weights',
                                                         r_model_path='Weights/rnet_Weights',
                                                         o_model_path='Weights/onet_Weights')

                    check = Face_alignment(check, default_square=True, landmarks=landmarks)
                    return check[0]
            except Exception as e:
                # print(img_path)
                # print(e)
                pass
for root, dirs, files in os.walk(parent_directory):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(file)
            image_path = os.path.join(root, file)
            aligned_face = detec_and_align(image_path)
            if aligned_face is not None:
                # Lấy tên người từ thư mục con
                person_name = os.path.basename(os.path.dirname(image_path))

                # Tạo thư mục con trong thư mục output
                output_directory =""
                output_directory = os.path.join(op, person_name)
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)

                # Lưu ảnh đã align vào thư mục output
                output_path = os.path.join(output_directory, file)
                cv2.imwrite(output_path, aligned_face)


