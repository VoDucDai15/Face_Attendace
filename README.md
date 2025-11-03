# 🎯 HỆ THỐNG NHẬN DIỆN KHUÔN MẶT  
### ỨNG DỤNG TRONG THỐNG KÊ LƯU LƯỢNG SINH VIÊN RA VÀO SỰ KIỆN

---

## 🧩 1. Lý do chọn đề tài

- **Hạn chế của phương pháp điểm danh thủ công:**
  - Tốn thời gian, dễ sai sót.
  - Không phù hợp với các sự kiện có quy mô lớn.

- **Nhu cầu công nghệ hóa quy trình điểm danh:**
  - Tăng hiệu quả, tiết kiệm thời gian, đảm bảo minh bạch.

- **Ứng dụng công nghệ nhận diện khuôn mặt:**
  - Tự động hóa, độ chính xác cao, góp phần nâng cao hình ảnh hiện đại cho nhà trường.

---

## 🎯 2. Mục tiêu và nhiệm vụ

### Mục tiêu
- Tự động hóa quy trình điểm danh.
- Đảm bảo tính **chính xác**, **công bằng**, **an toàn dữ liệu**.
- Quản lý dữ liệu hiệu quả và trực quan.

### Nhiệm vụ
- Nghiên cứu và áp dụng thuật toán nhận diện khuôn mặt hiện đại (FaceNet, MTCNN).
- Xây dựng hệ thống giao diện quản lý và thống kê.
- Kiểm tra và đánh giá độ chính xác trên dữ liệu thực tế.

---

## 🧪 3. Phạm vi và đối tượng nghiên cứu
- **Đối tượng:** Sinh viên trường Đại học Công Thương TP.HCM (HUIT) tham gia sự kiện.  
- **Phạm vi:**
  - Dữ liệu khuôn mặt.
  - Thời gian và môi trường triển khai.
  - Tính năng: điểm danh, thống kê, hiển thị kết quả.

---

## 🛠️ 4. Công nghệ và phương pháp thực hiện

### 🧠 Ngôn ngữ & Thư viện chính
| Công nghệ | Vai trò |
|------------|----------|
| **Python** | Ngôn ngữ chính, hỗ trợ xử lý ảnh và giao diện. |
| **OpenCV** | Phát hiện khuôn mặt và xử lý ảnh. |
| **MTCNN** | Phát hiện khuôn mặt đa điểm mốc (landmark) chính xác cao. |
| **FaceNet** | Trích xuất vector đặc trưng (embedding) để nhận diện khuôn mặt. |
| **MongoDB** | Lưu trữ thông tin sinh viên, sự kiện và dữ liệu điểm danh. |
| **Tkinter** | Xây dựng giao diện trực quan và thân thiện. |

---

## ⚙️ 5. Quy trình thực hiện

1. **Thu thập dữ liệu:**  
   - Ghi lại ảnh khuôn mặt sinh viên (≥5 ảnh mỗi người, nhiều góc độ).
   - Lưu trữ ảnh theo mã số sinh viên (MSSV).

2. **Huấn luyện mô hình:**  
   - Sử dụng FaceNet để trích xuất vector nhúng (embedding).  
   - Lưu embedding vào MongoDB để tra cứu.

3. **Nhận diện và điểm danh:**  
   - Mỗi lần camera quét, hệ thống so sánh vector mới với database.  
   - Ghi nhận trạng thái **Check-in/Check-out** và hiển thị thông tin sinh viên.

4. **Thống kê và báo cáo:**  
   - Xuất file CSV lưu dữ liệu điểm danh.  
   - Thống kê lưu lượng sinh viên ra/vào theo thời gian thực.

---

## 📈 6. Kết quả và đánh giá

| Tiêu chí | Kết quả đạt được |
|-----------|------------------|
| **Độ chính xác** | ~97.5% (FaceNet + MTCNN) |
| **Tốc độ xử lý** | < 1.5 giây/khuôn mặt |
| **Độ tin cậy** | Ghi nhận chính xác trạng thái điểm danh, không nhầm lẫn |
| **Giao diện** | Thân thiện, mượt mà, dễ dùng cho người không chuyên |

### Hệ thống bao gồm:
- Giao diện quét khuôn mặt & điểm danh tự động.  
- Quản lý thông tin sinh viên & sự kiện.  
- Xuất báo cáo thống kê.

---

## ⚠️ 7. Khó khăn và hướng khắc phục

| Vấn đề | Hướng giải quyết |
|---------|------------------|
| Ảnh hưởng ánh sáng, góc chụp | Bổ sung dữ liệu đa dạng, huấn luyện thêm. |
| Dữ liệu lớn, truy xuất chậm | Dùng MongoDB + Filebase để tối ưu. |
| Bảo mật dữ liệu khuôn mặt | Dự kiến áp dụng mã hóa dữ liệu (chưa triển khai). |

---

## 🚀 8. Ứng dụng và triển vọng

### Ứng dụng thực tế
- **Điểm danh sự kiện, lớp học, chấm công doanh nghiệp.**  
- Tự động hóa quy trình, giảm gian lận, tăng hiệu suất.

### Triển vọng phát triển
1. Tích hợp AI cải thiện độ chính xác & tốc độ.  
2. Ứng dụng IoT để giám sát ra vào thông minh.  
3. Phát triển hướng **thành phố thông minh** hoặc **an ninh khu vực**.  
4. Mở mã nguồn cho cộng đồng phát triển mở rộng.

---

## 🧾 9. Kết luận
Ứng dụng nhận diện khuôn mặt là một giải pháp **hiện đại, tự động và chính xác**, giúp **nâng cao hiệu quả quản lý**, **tiết kiệm thời gian**, và **hạn chế gian lận điểm danh**.  
Với AI & IoT, hệ thống có tiềm năng mở rộng mạnh mẽ cho các lĩnh vực như **an ninh, chấm công, quản lý lớp học**, phù hợp với xu hướng **chuyển đổi số 4.0**.

---

## 💻 Hướng dẫn cài đặt (gợi ý)

```bash
git clone https://github.com/<username>/NhanDienKhuonMat.git
cd NhanDienKhuonMat
pip install -r requirements.txt
python src/main.py
