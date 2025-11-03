import tkinter as tk
from PIL import Image, ImageTk  # Import PIL để xử lý hình ảnh
import subprocess
from pymongo import MongoClient

# Tạo hàm để chạy file Scan2.py
def run_scan():
    subprocess.run(["python", "Scan2.py"])  # Chạy Scan2.py

# Tạo hàm để chạy file facenet_train.py
def run_train():
    subprocess.run(["python", "facenet_train.py"])  # Chạy facenet_train.py

# Tạo hàm để chạy file facenet_test.py
def run_start():
    subprocess.run(["python", "DiemDanhCheck-In.py"])  # Chạy facenet_test.py

# Tạo cửa sổ chính
window = tk.Tk()
window.title("FaceNet Application")
window.geometry("450x450")  # Thay đổi kích thước cửa sổ
window.configure(bg="#f0f0f0")  # Màu nền

# Thêm tiêu đề
title_label = tk.Label(window, text="FaceNet Application", font=("Arial", 20), bg="#f0f0f0", fg="#333")
title_label.pack(pady=20)

# Tải hình ảnh
image_path = "../asset/face-recognition.png"  # Đường dẫn đến hình ảnh của bạn
image = Image.open(image_path)  # Mở hình ảnh
image = image.resize((100, 100), Image.LANCZOS)  # Thay đổi kích thước hình ảnh (nếu cần)
photo = ImageTk.PhotoImage(image)  # Chuyển đổi hình ảnh thành định dạng có thể hiển thị trong Tkinter

# Thêm hình ảnh vào cửa sổ
image_label = tk.Label(window, image=photo, bg="#f0f0f0")
image_label.pack(pady=10)  # Thêm khoảng cách trên hình ảnh

# Tạo 3 nút
btn_scan = tk.Button(window, text="Scan", command=run_scan, font=("Arial", 14), bg="#4CAF50", fg="white", padx=10, pady=5)
btn_scan.pack(pady=10)

btn_train = tk.Button(window, text="Train", command=run_train, font=("Arial", 14), bg="#2196F3", fg="white", padx=10, pady=5)
btn_train.pack(pady=10)

btn_start = tk.Button(window, text="Start", command=run_start, font=("Arial", 14), bg="#FF9800", fg="white", padx=10, pady=5)
btn_start.pack(pady=10)

# Chạy giao diện
window.mainloop()
