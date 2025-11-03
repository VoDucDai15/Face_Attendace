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
events = db['sukien']
# Load embedding vectors từ file
student_embeddings = np.load('embeddings.npy', allow_pickle=True).item()

# Khởi tạo mô hình FaceNet và MTCNN
embedder = FaceNet()
detector = MTCNN()

# Biến toàn cục cho treeview và danh sách đã điểm danh
treeview = None
recognized_students = set()
attendance_data = []

def create_embedding(face_img):
    """Tạo embedding vector từ ảnh khuôn mặt."""
    face_img_resized = cv2.resize(face_img, (160, 160))
    return embedder.embeddings([face_img_resized])[0]

def update_attendance(mssv, time_check_in, trang_thai):
    """Cập nhật thông tin điểm danh vào cơ sở dữ liệu."""
    result = db.sukien.update_one(
        {'mask': '12DHTH13'},
        {
            '$set': {
                f'dssinhvien_thamgia.$[elem].tgiancheck_in': time_check_in,
                f'dssinhvien_thamgia.$[elem].trangthai_chkin': trang_thai
            }
        },
        array_filters=[{'elem.mssv': mssv}]
    )
    if result.modified_count > 0:
        print("Cập nhật thành công")
    else:
        print("Không tìm thấy hoặc không cần cập nhật")

# Hàm cập nhật thời gian hiện tại
def update_time():
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    label_time.config(text=f"Thời gian hiện tại: {current_time}")
    root.after(1000, update_time)


# Hàm load thông tin từ MSSV và cập nhật label
def load_data_by_mssv(mssv):
    global recognized_students, label_name, label_mssv, student_image_label
    if mssv in recognized_students:
        return  # Nếu MSSV đã được nhận diện, không cần cập nhật lại

    student = students.find_one({"mssv": mssv})
    if student:
        # Cập nhật thông tin sinh viên
        label_name.config(text=f"Họ tên: {student['hoten']}")
        label_mssv.config(text=f"MSSV: {student['mssv']}")

        # Load và hiển thị hình ảnh
        image_path = os.path.join("../data/dataset", mssv, f"{mssv}_1.jpg")
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

        # Cập nhật thời gian và trạng thái điểm danh vào database
        check_in_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        update_attendance(mssv, check_in_time, 'Đã điểm danh vào')

        # Thông tin đầy đủ được lưu trữ và cập nhật vào TreeView
        date, time = check_in_time.split()
        save_attendance(student['hoten'], student['lop'], student['mssv'], date, time, "Đã điểm danh vào")

        recognized_students.add(mssv)
        play_sound(moivao_sound)  # Phát âm thanh "mở vào"
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


# Hàm mở cửa sổ chứa Treeview
def open_attendance_window():
    attendance_window = tk.Toplevel(root)
    attendance_window.title("Thông tin sinh viên đã điểm danh")
    attendance_window.geometry("1200x500")

    global treeview
    treeview = ttk.Treeview(attendance_window, columns=("Họ tên", "Lớp", "MSSV", "Ngày vào", "Giờ vào", "Trạng thái"),
                            show="headings")
    treeview.heading("Họ tên", text="Họ tên")
    treeview.heading("Lớp", text="Lớp")
    treeview.heading("MSSV", text="MSSV")
    treeview.heading("Ngày vào", text="Ngày vào")
    treeview.heading("Giờ vào", text="Giờ vào")
    treeview.heading("Trạng thái", text="Trạng thái")
    treeview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Hiển thị số lượng sinh viên đã checkin
    total_students = students.count_documents({})
    checked_in_students = len(recognized_students)
    label_attendance_info = ttk.Label(attendance_window,
                                      text=f"Số sinh viên đã điểm danh: {checked_in_students}/{total_students}")
    label_attendance_info.pack(pady=5)

    # Lấy dữ liệu điểm danh từ database
    attendance_data = []
    for event in events.find():
        if event['mask'] == '12DHTH13':
            for student in event['dssinhvien_thamgia']:
                if student['tgiancheck_in']:
                    date, time = student['tgiancheck_in'].split()
                    attendance_data.append((student['hoten'], student['lop'], student['mssv'], date, time, student['trangthai_chkin']))

    # Hiển thị dữ liệu vào TreeView
    for data in attendance_data:
        treeview.insert("", "end", values=data)

    # Tạo nút "Xuất file"
    export_button = ttk.Button(attendance_window, text="Xuất file",
                               command=lambda: root.after(10, export_attendance_to_csv))
    export_button.pack(pady=10)

    # Tạo nút "Thêm sinh viên vắng"
    add_absent_button = ttk.Button(attendance_window, text="Xong", command=add_absent_students)
    add_absent_button.pack(pady=5)


# Hàm xuất thông tin điểm danh ra file CSV
def export_attendance_to_csv():
    global treeview
    if not attendance_data:
        messagebox.showwarning("Cảnh báo", "Chưa có thông tin điểm danh để xuất!")
        return

    try:
        with open("../data/diemdanh_data.csv", "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Họ tên", "Lớp", "MSSV", "Ngày vào", "Giờ vào", "Trạng thái"])
            for item in treeview.get_children():
                values = treeview.item(item, 'values')
                writer.writerow(values)
        messagebox.showinfo("Thông báo", "Xuất file thành công!")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi xuất file: {e}")


# Hàm thêm sinh viên vắng vào TreeView
def add_absent_students():
    global treeview, attendance_data
    for student in students.find():
        if student['mssv'] not in recognized_students:
            # Thêm sinh viên vào TreeView nếu chưa có
            current_date = datetime.now().strftime('%Y-%m-%d')
            treeview.insert("", "end",
                            values=(student['hoten'], student['lop'], student['mssv'], current_date, None, "Vắng"))
            attendance_data.append((student['hoten'], student['lop'], student['mssv'], current_date, None, "Vắng"))


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
            face_img = gray[y:y + h, x:x + w]
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
                cv2.putText(frame, f"MSSV: {best_match}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                current_time = time.time()
                if current_time - last_unknown_sound_time >= 3:  # Kiểm tra khoảng cách thời gian
                    play_sound(khongxacdinh_sound)  # Phát âm thanh "không xác định"
                    last_unknown_sound_time = current_time  # Cập nhật thời gian phát âm thanh cuối cùng

                # Vẽ khung chữ nhật và "Unknown" lên hình
                cv2.putText(frame, f"Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

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
moivao_sound = os.path.join("../asset/Audio", "moivao.mp3")
khongxacdinh_sound = os.path.join("../asset/Audio", "khongxacdinh.mp3")


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
root.geometry("1000x600")

# Frame chính
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Phần camera
camera_frame = tk.Frame(main_frame)
camera_frame.pack(side=tk.LEFT, padx=10, pady=10)

camera_label = tk.Label(camera_frame)
camera_label.pack()

# Phần bên phải: Hiển thị thông tin sinh viên
info_frame = tk.Frame(main_frame)
info_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Label hiển thị ảnh
student_image_label = ttk.Label(info_frame)
student_image_label.grid(row=0, column=0, padx=5, pady=5)

# Chỉ hiển thị MSSV và Họ tên
label_name = ttk.Label(info_frame, text="Họ tên: ", font=('Arial', 12))
label_name.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)

label_mssv = ttk.Label(info_frame, text="MSSV: ", font=('Arial', 12))
label_mssv.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)

# Nút đóng chương trình
close_button = ttk.Button(root, text="Đóng", command=close_app)
close_button.pack(side=tk.RIGHT, padx=10, pady=10)

# Nút mở cửa sổ điểm danh
attendance_button = ttk.Button(root, text="Xem thông tin điểm danh", command=open_attendance_window)
attendance_button.pack(side=tk.LEFT, padx=10, pady=10)

# Bắt đầu nhận diện khuôn mặt
recognize_thread = threading.Thread(target=recognize_face)
recognize_thread.daemon = True
recognize_thread.start()

# Hiển thị thời gian hiện tại
label_time = ttk.Label(root, text="Thời gian hiện tại: ")
label_time.pack(pady=5)
update_time()

# Chạy vòng lặp giao diện
root.mainloop()