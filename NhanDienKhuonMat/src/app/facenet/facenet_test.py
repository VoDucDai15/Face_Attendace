import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from pymongo import MongoClient
import csv
from datetime import datetime
from keras_facenet import FaceNet
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
import threading
import pygame  # Import thư viện pygame
import time  # Import thư viện time

# Kết nối tới MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['nhandienkhuonmat']
students = db['sinhvien']

# Load embedding vectors từ file
student_embeddings = np.load('embeddings.npy', allow_pickle=True).item()

# Khởi tạo mô hình FaceNet và MTCNN
embedder = FaceNet()
detector = MTCNN()

# Hàm tạo embedding vector từ ảnh
def create_embedding(face_img):
    face_img_resized = cv2.resize(face_img, (160, 160))
    return embedder.embeddings([face_img_resized])[0]

# Biến toàn cục cho treeview và danh sách đã điểm danh
treeview = None
recognized_students = set()
attendance_data = []

# Hàm cập nhật thời gian hiện tại
def update_time():
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    label_time.config(text=f"Thời gian hiện tại: {current_time}")
    root.after(1000, update_time)

# Hàm load thông tin từ MSSV và cập nhật label
def load_data_by_mssv(mssv):
    global recognized_students, label_name, label_mssv, student_image_label, treeview, has_played_moivao
    
    student = students.find_one({"mssv": mssv})
    if student:
        # Cập nhật thông tin sinh viên
        label_name.config(text=f"Họ tên: {student['hoten']}")
        label_mssv.config(text=f"MSSV: {student['mssv']}")

        # Load và hiển thị hình ảnh
        image_path = os.path.join("../../../data/dataset", mssv, f"{mssv}_1.jpg")
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                img = img.resize((150, 150))
                imgtk = ImageTk.PhotoImage(img)
                student_image_label.imgtk = imgtk  # Lưu trữ hình ảnh để tránh bị xóa bởi GC
                student_image_label.config(image=imgtk)
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi load ảnh: {e}")
        else:
            student_image_label.config(image="")  # Xóa ảnh nếu không tìm thấy

        # Kiểm tra xem sinh viên đã điểm danh chưa
        if mssv not in recognized_students:
            # Thông tin đầy đủ được lưu trữ và cập nhật vào TreeView
            check_in_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            date, time = check_in_time.split()

            # Lưu thông tin điểm danh đầy đủ vào treeview
            save_attendance(student['hoten'], student['lop'], student['mssv'], date, time, "Đã điểm danh vào")
            
            # **Cập nhật Treeview**
            treeview.insert("", 0, values=(student['hoten'], student['lop'], student['mssv'], date, time, "Đã điểm danh vào"))
            
            recognized_students.add(mssv)
            # Phát âm thanh "mời vào" và reset biến cờ
            if not has_played_moivao:
                play_sound(moivao_sound)  
                has_played_moivao = True
        else:
            # Nếu sinh viên đã điểm danh, reset biến cờ để cho phép phát âm thanh lần sau
            has_played_moivao = False
    else:
        messagebox.showwarning("Cảnh báo", "Không tìm thấy sinh viên với MSSV này")

# Hàm lưu thông tin điểm danh
def save_attendance(hoten, lop, mssv, date, time, status):
    global attendance_data
    # Tìm vị trí của sinh viên trong attendance_data
    for i, data in enumerate(attendance_data):
        if data[2] == mssv:
            attendance_data[i] = (hoten, lop, mssv, date, time, status)
            break
    else:
        # Nếu không tìm thấy, thêm dòng mới
        attendance_data.append((hoten, lop, mssv, date, time, status))

# Hàm nhận diện khuôn mặt
def recognize_face():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    last_unknown_sound_time = time.time()  # Lưu thời gian phát âm thanh "không xác định" lần cuối

    def update_frame(last_unknown_sound_time):  # Truyền biến vào hàm
        global recognized_students, label_name, label_mssv
        ret, frame = cap.read()
        if not ret:
            return
        
        # Chuyển đổi sang ảnh màu
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Tạo embedding cho khuôn mặt
            face_img = gray[y:y+h, x:x+w]
            face_embedding = create_embedding(face_img)

            # Tìm sinh viên có embedding gần nhất
            best_match = None
            min_distance = float('inf')
            for mssv, known_embedding in student_embeddings.items():
                distance = cosine(face_embedding, known_embedding)
                if distance < min_distance:
                    min_distance = distance
                    best_match = mssv
            
            # Kiểm tra độ tin cậy
            if min_distance < 0.2:
                load_data_by_mssv(best_match)  # Cập nhật nhãn thông tin sinh viên
                # Vẽ khung chữ nhật và MSSV lên hình
                cv2.putText(frame, f"MSSV: {best_match}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            else:
                current_time = time.time()
                if current_time - last_unknown_sound_time >= 3:  # Kiểm tra khoảng cách thời gian
                    play_sound(khongxacdinh_sound)  # Phát âm thanh "không xác định"
                    last_unknown_sound_time = current_time  # Cập nhật thời gian phát âm thanh cuối cùng
                
                # Vẽ khung chữ nhật và "Unknown" lên hình
                cv2.putText(frame, f"Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))
        camera_label.imgtk = imgtk
        camera_label.config(image=imgtk)
        camera_label.after(10, lambda: update_frame(last_unknown_sound_time))  # Gọi lại hàm update_frame

    # Gọi hàm update_frame() và truyền biến last_unknown_sound_time
    update_frame(last_unknown_sound_time)

# Hàm để đóng chương trình
def close_app():
    root.quit()
    root.destroy()

# Khởi tạo pygame
pygame.mixer.init()

# Đường dẫn đến file âm thanh
moivao_sound = os.path.join("../../../asset/Audio", "moivao.mp3")
khongxacdinh_sound = os.path.join("../../../asset/Audio", "khongxacdinh.mp3")

# Hàm phát âm thanh
def play_sound(sound_file):
    try:
        sound = pygame.mixer.Sound(sound_file)
        sound.play()
    except pygame.error as e:
        print(f"Lỗi phát âm thanh: {e}")

# Cửa sổ chính của ứng dụng
root = tk.Tk()
root.title("Hệ thống nhận diện khuôn mặt")
root.geometry("1000x700")  

# Frame chính để chứa camera và thông tin
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Phần camera
camera_frame = tk.Frame(main_frame)
camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Giảm kích thước khung camera
camera_label = tk.Label(camera_frame, width=400, height=300) 
camera_label.pack()

# Frame hiển thị thông tin sinh viên
info_frame = tk.Frame(main_frame)
info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

label_name = tk.Label(info_frame, text="Họ tên: ")
label_name.pack()
label_name.config(font=("Arial", 16))  # Thay đổi font chữ thành Arial, kích thước 16

label_mssv = tk.Label(info_frame, text="MSSV: ")
label_mssv.pack()
label_mssv.config(font=("Arial", 16))

# Thêm label để hiển thị hình ảnh sinh viên
student_image_label = tk.Label(info_frame)
student_image_label.pack()

# Phần TreeView để hiển thị danh sách sinh viên
tree_frame = tk.Frame(root)  
tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

columns = ("Họ tên", "Lớp", "MSSV", "Ngày vào", "Giờ vào", "Trạng thái")
treeview = ttk.Treeview(tree_frame, columns=columns, show="headings")

for col in columns:
    treeview.heading(col, text=col)

treeview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Hiển thị thời gian hiện tại ở cuối cửa sổ
label_time = ttk.Label(root, text="Thời gian hiện tại: ")
label_time.pack(pady=5)
update_time()

# Biến cờ để kiểm tra phát âm thanh "mời vào"
has_played_moivao = False

# Tạo luồng mới để chạy hàm nhận diện khuôn mặt
recognition_thread = threading.Thread(target=recognize_face)
recognition_thread.daemon = True
recognition_thread.start()

# Đảm bảo đóng chương trình khi người dùng thoát
root.protocol("WM_DELETE_WINDOW", close_app)

# Chạy vòng lặp giao diện
root.mainloop()