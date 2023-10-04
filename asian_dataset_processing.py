import os
import cv2
from align_trans import Face_alignment
import torch
from torchvision import transforms as trans
# Đường dẫn đến thư mục mẹ chứa tất cả các thư mục con
parent_directory = '/path/to/parent_directory'
from MTCNN import create_mtcnn_net
# Đường dẫn đến thư mục lưu kết quả sau khi alignment
output_directory = '/path/to/output_directory'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
def detec_and_align():
    try:
        img = cv2.imread(image_path)

        if img.shape != None:
            bboxes, landmarks = create_mtcnn_net(img, 20, device,
                                                 p_model_path='Weights/pnet_Weights',
                                                 r_model_path='Weights/rnet_Weights',
                                                 o_model_path='Weights/onet_Weights')
    except Exception as e:
        print(e)
        img = Face_alignment(img, default_square=True, landmarks=landmarks)
        return img[0]
for root, dirs, files in os.walk(parent_directory):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            aligned_face = detec_and_align(image)
            if aligned_face is not None:
                # Lấy tên người từ thư mục con
                person_name = os.path.basename(os.path.dirname(image_path))

                # Tạo thư mục con trong thư mục output
                output_directory = os.path.join(output_directory, person_name)
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)

                # Lưu ảnh đã align vào thư mục output
                output_path = os.path.join(output_directory, file)
                cv2.imwrite(output_path, aligned_face)





