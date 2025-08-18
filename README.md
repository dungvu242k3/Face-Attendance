# 👨‍💼 Face Attendance System

🚀 Hệ thống **chấm công bằng nhận diện khuôn mặt** được xây dựng với **YOLO + FaceNet + Liveness Detection**, lưu trữ dữ liệu bằng **PostgreSQL**, bao gồm:

- 📲 **User App (app_user)**: Nhân viên chấm công (check-in / check-out) bằng camera.
- 🖥 **Admin Dashboard (app_admin)**: Quản lý nhân viên, theo dõi lịch sử chấm công, thêm nhân viên bằng upload ảnh hoặc webcam.

---

## ✨ Tính năng nổi bật

### 🧠 Công nghệ
- **YOLO** → Phát hiện khuôn mặt trong hình ảnh / video.
- **FaceNet** → Sinh ra **embedding 512 chiều** từ khuôn mặt.
- **Liveness Detection** → Ngăn chặn giả mạo bằng ảnh / video.
- **PostgreSQL** → Lưu trữ embedding, thông tin nhân viên và lịch sử chấm công.
- **Không lưu ảnh gốc** → chỉ giữ embedding ⇒ nhẹ & tối ưu.

---

### 📊 Admin Dashboard (`app_admin`)
- ✅ **Tổng quan (Dashboard)**: số nhân viên, số lượt chấm công hôm nay, đi muộn, và lịch sử gần nhất.
- 👨‍💻 **Quản lý nhân viên**: thêm (ảnh upload / webcam), sửa, xoá.
- 🕒 **Lịch sử chấm công**: xem theo nhân viên, loại (check-in / check-out).

### 📱 User App (`app_user`)
- 📸 Nhân viên chấm công trực tiếp qua camera.
- 🔒 Hệ thống kiểm tra **liveness** trước khi ghi nhận.
- 📝 Tự động lưu vào **attendance_logs_512**.

---

## 🗂 Cấu trúc Database (PostgreSQL)

| Bảng                  | Mô tả                                                    |
|-----------------------|----------------------------------------------------------|
| **employees**         | Thông tin nhân viên (id, name, email, phone, position)  |
| **face_embeddings_512** | Embedding khuôn mặt (512 chiều) + liên kết employee_id |
| **attendance_logs_512** | Lịch sử chấm công (thời gian, loại checkin/checkout)   |

---

## ⚙️ Cài đặt & Khởi chạy

### 1. Clone dự án
```bash
git clone https://github.com/dungvu242k3/Face-Attendance.git
cd Face-Attendance
````

### 2. Tạo môi trường & cài thư viện

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Cấu hình `.env`

Tạo file `.env` ở thư mục gốc:

```ini
SECRET_KEY=your_secret_key
DB_NAME=face_db
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
UPLOAD_FOLDER=uploads
USE_MOCK_DETECTOR=false
```

### 4. Khởi tạo Database

```bash
psql -U postgres -d face_db -f init_db.sql
```

### 5. Chạy ứng dụng

```bash
python app_admin.py
```

Mở trình duyệt tại:

* 🌐 `http://localhost:5000/admin/dashboard` → Dashboard
* 🌐 `http://localhost:5000/admin/add_employee` → Thêm nhân viên
* 🌐 `http://localhost:5000/admin/employees` → Quản lý nhân viên
* 🌐 `http://localhost:5000/user/checkin` → Chấm công

---

## 🛠 Kiến trúc Hệ thống

```
Camera → YOLO → FaceNet → Liveness Detection → Embedding (512D) → PostgreSQL
```

* Embedding được so sánh với DB để xác định danh tính.
* Nếu hợp lệ → ghi log check-in / check-out.

---

## 🚀 Hướng phát triển tương lai

* 🐳 Docker hoá toàn bộ hệ thống.
* 📱 Mobile App cho nhân viên (API chấm công).
* 🤖 Liveness Detection nâng cao với Deep Learning (CNN / Transformer).
* 📊 Dashboard nâng cấp với biểu đồ thống kê.

---

## 👨‍💻 Tác giả

* Dự án bởi **[dungvu242k3](https://github.com/dungvu242k3)**


