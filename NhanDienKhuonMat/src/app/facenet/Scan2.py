import tkinter as tk
from tkinter import messagebox
import cv2
import os
from pymongo import MongoClient
from gtts import gTTS
import time
import pygame

# Kết nối MongoDB (Bạn có thể bỏ phần này nếu không cần lưu vào database)
client = MongoClient('mongodb://localhost:27017/')  
db = client['nhandienkhuonmat']  
students = db['sinhvien']  

# Khởi tạo ứng dụng tkinter
root = tk.Tk()
root.title("Thu thập dữ liệu khuôn mặt sinh viên")

# Biến toàn cục cho MSSV
current_mssv = ""

# Thư mục lưu dataset
dataset_dir = '../../../data/dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)


# Hàm phát âm thanh từ file có sẵn bằng pygame
def speak(file_path):
    full_path = os.path.join('../../../asset/Audio', file_path + '.mp3').replace("\\", "/")
    if os.path.exists(full_path):
        pygame.mixer.init()  # Khởi tạo pygame mixer
        pygame.mixer.music.load(full_path)  # Load file âm thanh
        pygame.mixer.music.play()  # Phát âm thanh
        while pygame.mixer.music.get_busy():  # Đợi cho đến khi âm thanh phát xong
            pygame.time.Clock().tick(10)
    else:
        print(f"File {file_path} không tồn tại.")


# Hàm kiểm tra MSSV trong CSDL
def check_student():
    global current_mssv
    mssv = mssv_entry.get()

    # Kiểm tra độ dài MSSV
    if len(mssv) != 10:
        messagebox.showerror("Lỗi", "MSSV phải có đúng 10 ký tự.")
        reset_form()
        return

    # Nếu MSSV hợp lệ, kiểm tra thông tin sinh viên trong cơ sở dữ liệu
    student = students.find_one({"mssv": mssv})
    if student:
        # Cập nhật thông tin sinh viên vào giao diện
        hoten_entry.config(state="normal")
        hoten_entry.delete(0, tk.END)
        hoten_entry.insert(0, student.get("hoten", ""))
        hoten_entry.config(state="readonly")

        lop_entry.config(state="normal")
        lop_entry.delete(0, tk.END)
        lop_entry.insert(0, student.get("lop", ""))
        lop_entry.config(state="readonly")

        current_mssv = mssv
    else:
        messagebox.showerror("Lỗi", "Sinh viên chưa đăng ký tham gia sự kiện.")
        reset_form()

# Hàm chụp ảnh và lưu vào CSDL và dataset
def capture_images():
    global current_mssv
    mssv = current_mssv
    if mssv:
        student_images_dir = os.path.join(dataset_dir, mssv)

        # Kiểm tra nếu đã có hình ảnh khuôn mặt
        if os.path.exists(student_images_dir) and len(os.listdir(student_images_dir)) > 0:
            if not messagebox.askyesno("Thông báo", "Sinh viên đã có hình ảnh khuôn mặt. Bạn có muốn chụp lại dữ liệu không?"):
                return  # Nếu không muốn chụp lại, thoát khỏi hàm

        # Nếu không có hình ảnh hoặc người dùng muốn chụp lại, thực hiện quá trình chụp
        if not os.path.exists(student_images_dir):
            os.makedirs(student_images_dir)

        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        speak("batdau")  # Thông báo khi bắt đầu

        total_images = 50
        current_images = 0

        while current_images < total_images:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                current_images += 1
                face = gray[y:y+h, x:x+w]
                file_name_path = os.path.join(student_images_dir, f'{mssv}_{current_images}.jpg')
                cv2.imwrite(file_name_path, face)
                cv2.putText(frame, f"Saving {current_images}/{total_images}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Phát giọng nói cho các khoảng ảnh
            if current_images == 10:
                speak("nhintren")
            elif current_images == 20:
                speak("nhinduoi")
            elif current_images == 30:
                speak("nhintrai")
            elif current_images == 40:
                speak("nhinphai")

            cv2.imshow('Thu thap du lieu khuon mat', frame)
            time.sleep(0.1)  # Chờ 0.1 giây

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        speak("hoanthanh")  # Thông báo khi hoàn thành
        messagebox.showinfo("Hoàn tất", f"Đã lưu {current_images} ảnh cho MSSV {mssv}")
        
        # Reset form sau khi chụp xong
        reset_form()
    else:
        messagebox.showerror("Lỗi", "Bạn chưa nhập MSSV.")
# Hàm reset form
def reset_form():
    global current_mssv
    current_mssv = ""
    mssv_entry.delete(0, tk.END)
    hoten_entry.config(state="normal")
    hoten_entry.delete(0, tk.END)
    hoten_entry.config(state="readonly")
    lop_entry.config(state="normal")
    lop_entry.delete(0, tk.END)
    lop_entry.config(state="readonly")

# Giao diện nhập thông tin
tk.Label(root, text="Nhập MSSV:").grid(row=0, column=0, padx=10, pady=5)
mssv_entry = tk.Entry(root)
mssv_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Button(root, text="Kiểm Tra", command=check_student).grid(row=0, column=2, padx=10, pady=5)

tk.Label(root, text="Kiểm tra thông tin").grid(row=1, column=0, columnspan=2, pady=(15, 5))

tk.Label(root, text="Họ Tên:").grid(row=2, column=0, padx=10, pady=5)
hoten_entry = tk.Entry(root, state="readonly")
hoten_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Lớp:").grid(row=3, column=0, padx=10, pady=5)
lop_entry = tk.Entry(root, state="readonly")
lop_entry.grid(row=3, column=1, padx=10, pady=5)

chup_btn = tk.Button(root, text="Bắt đầu chụp", command=capture_images)
chup_btn.grid(row=5, column=1, padx=10, pady=10)

root.geometry("350x300")
root.mainloop()