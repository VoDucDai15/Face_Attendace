# 🎯 Hệ thống Nhận Diện Khuôn Mặt - Điểm Danh Sinh Viên

Dự án này là đồ án chuyên ngành, xây dựng hệ thống nhận diện khuôn mặt để điểm danh sinh viên ra vào sự kiện bằng Python và OpenCV.

## 🧠 Tính năng chính
- Nhận diện khuôn mặt sử dụng FaceNet.
- Kết nối MongoDB để lấy thông tin sinh viên.
- Giao diện tkinter để nhập MSSV và hiển thị thông tin.
- Lưu ảnh khuôn mặt theo MSSV và thứ tự.
- Kiểm tra và hiển thị hình ảnh đã có.

## 🗂️ Cấu trúc thư mục
- src/
- ├── app/
- │ ├── diem_danh/
- │ ├── facenet/
- │ └── utils/
- ├── data/
- ├── assets/
- └── tests/

## 🚀 Cách chạy chương trình
1. Tạo môi trường ảo:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
2. Cài đặt thư viện:
   pip install -r requirements.txt
3. Chạy ứng dụng:
   python src/main.py

## 📚 Công nghệ:
+ Python (OpenCV, TensorFlow, Tkinter, Pymongo)
+ MongoDB
+ FaceNet Model
