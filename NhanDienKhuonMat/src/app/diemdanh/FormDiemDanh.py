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
    global recognized_students, label_name, label_class, label_mssv, label_date, label_check_in, label_status
    if mssv in recognized_students:
        return  # Nếu MSSV đã được nhận diện, không cần cập nhật lại

    student = students.find_one({"mssv": mssv})
    if student:
        # Cập nhật thông tin sinh viên trên các nhãn
        label_name.config(text=f"Họ tên: {student['hoten']}")
        label_class.config(text=f"Lớp: {student['lop']}")
        label_mssv.config(text=f"MSSV: {student['mssv']}")

        # Cập nhật thời gian nhận diện và trạng thái
        check_in_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        date, time = check_in_time.split()
        label_date.config(text=f"Ngày vào: {date}")
        label_check_in.config(text=f"Giờ vào: {time}")
        label_status.config(text="Trạng thái: Đã điểm danh vào")

        # Lưu thông tin điểm danh
        save_attendance(student['hoten'], student['lop'], student['mssv'], date, time, "Đã điểm danh vào")
        recognized_students.add(mssv)
    else:
        messagebox.showwarning("Cảnh báo", "Không tìm thấy sinh viên với MSSV này")

# Hàm lưu thông tin điểm danh
def save_attendance(hoten, lop, mssv, date, time, status, check_out_time=None):
    global attendance_data
    # Tìm vị trí của sinh viên trong attendance_data
    for i, data in enumerate(attendance_data):
        if data[2] == mssv:
            attendance_data[i] = (hoten, lop, mssv, date, time, check_out_time, status)
            break
    else:
        # Nếu không tìm thấy, thêm dòng mới
        attendance_data.append((hoten, lop, mssv, date, time, check_out_time, status))

# Hàm mở cửa sổ chứa Treeview
def open_attendance_window():
    attendance_window = tk.Toplevel(root)
    attendance_window.title("Thông tin sinh viên đã điểm danh")
    attendance_window.geometry("1200x500")  

    global treeview
    treeview = ttk.Treeview(attendance_window, columns=("Họ tên", "Lớp", "MSSV", "Ngày vào", "Giờ vào", "Giờ ra", "Trạng thái"), show="headings")
    treeview.heading("Họ tên", text="Họ tên")
    treeview.heading("Lớp", text="Lớp")
    treeview.heading("MSSV", text="MSSV")
    treeview.heading("Ngày vào", text="Ngày vào")
    treeview.heading("Giờ vào", text="Giờ vào")
    treeview.heading("Giờ ra", text="Giờ ra")
    treeview.heading("Trạng thái", text="Trạng thái")
    treeview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Hiển thị số lượng sinh viên đã checkin
    total_students = students.count_documents({})
    checked_in_students = len(recognized_students)
    label_attendance_info = ttk.Label(attendance_window, text=f"Số sinh viên đã điểm danh: {checked_in_students}/{total_students}")
    label_attendance_info.pack(pady=5)

    for data in attendance_data:
        treeview.insert("", "end", values=data)

    # Tạo nút "Xuất file"
    export_button = ttk.Button(attendance_window, text="Xuất file", command=lambda: root.after(100, export_attendance_to_csv))
    export_button.pack(pady=10)

# Hàm xuất thông tin điểm danh ra file CSV
def export_attendance_to_csv():
    global treeview
    if not attendance_data:  
        messagebox.showwarning("Cảnh báo", "Chưa có thông tin điểm danh để xuất!")
        return

    try:
        with open("../../../data/diemdanh_data.csv", "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Họ tên", "Lớp", "MSSV", "Ngày vào", "Giờ vào", "Giờ ra", "Trạng thái"])
            for item in treeview.get_children():
                values = treeview.item(item, 'values')
                writer.writerow(values)
        messagebox.showinfo("Thông báo", "Xuất file thành công!")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi xuất file: {e}")

# Hàm nhận diện khuôn mặt
def recognize_face():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def update_frame():
        global recognized_students, label_name, label_class, label_mssv, label_date, label_check_in, label_status
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
            cv2.putText(frame, f"MSSV: {best_match if best_match else 'Unknown'}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if best_match else (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))
        camera_label.imgtk = imgtk
        camera_label.config(image=imgtk)
        camera_label.after(10, update_frame)

    update_frame()

# Hàm để đóng chương trình
def close_app():
    root.quit()
    root.destroy()

# Hàm xử lý checkout
def checkout():
    # Lấy MSSV từ label_mssv (nếu đã nhận diện)
    mssv = label_mssv.cget("text").split(": ")[1] if "MSSV: " in label_mssv.cget("text") else None
    
    if mssv and mssv in recognized_students:
        # Lưu thông tin checkout
        check_out_time = datetime.now().strftime('%H:%M:%S')
        
        # Biến để kiểm tra xem đã cập nhật hay chưa
        updated = False
        
        # Cập nhật trạng thái và giờ ra cho sinh viên
        for i, data in enumerate(attendance_data):
            if data[2] == mssv:
                attendance_data[i] = (data[0], data[1], mssv, data[3], data[4], check_out_time, "Đã điểm danh ra")
                update_treeview_item(i, attendance_data[i])
                label_status.config(text="Trạng thái: Đã điểm danh ra")
                updated = True
                break
        
        if updated:
            # Xóa thông tin trên tất cả các label
            label_mssv.config(text="MSSV: ")
            label_name.config(text="Họ tên: ")
            label_class.config(text="Lớp: ")
            label_date.config(text="Ngày vào: ")
            label_check_in.config(text="Giờ vào: ")
            label_status.config(text="Trạng thái: ")
            
            # Xóa MSSV khỏi recognized_students
            recognized_students.remove(mssv)
        else:
            messagebox.showwarning("Cảnh báo", "Chưa có thông tin điểm danh cho sinh viên này.")

    else:
        messagebox.showwarning("Cảnh báo", "Chưa có sinh viên nào được điểm danh vào")

# Cập nhật nội dung dòng dữ liệu trong Treeview
def update_treeview_item(index, values):
    global treeview
    if treeview:
        treeview.item(treeview.get_children()[index], values=values)

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Điểm danh sinh viên")

# Phía trên: Hiển thị thời gian hiện tại
label_time = ttk.Label(root, text="", font=('Arial', 12))
label_time.pack(side=tk.TOP, pady=10)
update_time()

# Tạo khung cho camera và thông tin
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Phần bên trái: Cửa sổ camera (hình vuông)
camera_frame = tk.Frame(main_frame, width=300, height=300, bg="black")
camera_frame.pack(side=tk.LEFT, padx=10, pady=10)
camera_label = tk.Label(camera_frame)
camera_label.pack(fill=tk.BOTH, expand=True)

# Phần bên phải: Hiển thị thông tin sinh viên
info_frame = tk.Frame(main_frame)
info_frame.pack(side=tk.RIGHT, padx=10, pady=10)

label_name = ttk.Label(info_frame, text="Họ tên: ", font=('Arial', 12))
label_name.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

label_class = ttk.Label(info_frame, text="Lớp: ", font=('Arial', 12))
label_class.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)

label_mssv = ttk.Label(info_frame, text="MSSV: ", font=('Arial', 12))
label_mssv.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)

label_date = ttk.Label(info_frame, text="Ngày vào: ", font=('Arial', 12))
label_date.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)

label_check_in = ttk.Label(info_frame, text="Giờ vào: ", font=('Arial', 12))
label_check_in.grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)

label_status = ttk.Label(info_frame, text="Trạng thái: ", font=('Arial', 12))
label_status.grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)

# Phía dưới: Các nút điều khiển
control_frame = tk.Frame(root)
control_frame.pack(side=tk.BOTTOM, pady=10)

attendance_button = ttk.Button(control_frame, text="Xem thông tin đã điểm danh", command=open_attendance_window)
attendance_button.grid(row=0, column=0, padx=10)

checkout_button = ttk.Button(control_frame, text="Checkout", command=checkout)
checkout_button.grid(row=0, column=1, padx=10)

close_button = ttk.Button(control_frame, text="Đóng", command=close_app)
close_button.grid(row=0, column=2, padx=10)

# Hàm điểm danh vắng tự động
def mark_absent():
    global attendance_data
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    date, time = current_time.split()

    for student in students.find():
        mssv = student['mssv']
        if mssv not in recognized_students:
            save_attendance(student['hoten'], student['lop'], mssv, date, '', None, None, "Vắng")

# Khởi động camera và nhận diện khuôn mặt
def start_recognition():
    recognize_face()
    # Bắt đầu điểm danh vắng tự động sau 2 phút
    threading.Timer(1800, mark_absent).start()

# Khởi động camera và nhận diện khuôn mặt
start_recognition()

root.mainloop()