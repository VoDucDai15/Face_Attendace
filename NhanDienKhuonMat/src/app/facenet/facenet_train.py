import os
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
import cv2

# Đường dẫn đến folder chứa ảnh sinh viên, mỗi folder là MSSV của sinh viên
dataset_path = "../../../data/dataset"

# Khởi tạo mô hình FaceNet và MTCNN
embedder = FaceNet()
detector = MTCNN()

# Hàm tạo embedding vector từ ảnh
def create_embedding(face_img):
    face_img_resized = cv2.resize(face_img, (160, 160))
    return embedder.embeddings([face_img_resized])[0]

# Tạo dictionary lưu các embedding của từng sinh viên
student_embeddings = {}

# Duyệt qua từng folder trong dataset
print("Bắt đầu tạo embedding cho sinh viên...")
for mssv in os.listdir(dataset_path):
    student_folder = os.path.join(dataset_path, mssv)
    
    # Đảm bảo rằng mỗi folder chỉ chứa các file ảnh
    if os.path.isdir(student_folder):
        embeddings = []
        print(f"Đang xử lý folder: {mssv}")

        # Duyệt qua từng ảnh trong folder của sinh viên
        for img_file in os.listdir(student_folder):
            img_path = os.path.join(student_folder, img_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(rgb_img)
                
                if faces:
                    # Giả sử mỗi ảnh chỉ có 1 khuôn mặt
                    x, y, w, h = faces[0]['box']
                    face_img = rgb_img[y:y+h, x:x+w]
                    embedding = create_embedding(face_img)
                    embeddings.append(embedding)
                    print(f"Đã tạo embedding cho ảnh: {img_file}")

        if embeddings:
            # Lấy trung bình của các embedding nếu có nhiều ảnh
            mean_embedding = np.mean(embeddings, axis=0)
            student_embeddings[mssv] = mean_embedding
            print(f"Đã lưu embedding cho sinh viên: {mssv} (Tổng số ảnh: {len(embeddings)})")
        else:
            print(f"Không tìm thấy khuôn mặt trong folder: {mssv}")

print("Đã hoàn tất việc tạo embedding cho tất cả sinh viên.")

# Lưu lại embedding vectors vào file
np.save('embeddings.npy', student_embeddings)
print("Đã lưu embedding vectors vào file embeddings.npy.")
