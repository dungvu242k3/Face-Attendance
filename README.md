# Face Attendance

Hệ thống chấm công bằng nhận diện khuôn mặt gồm dashboard quản trị và ứng dụng camera cho nhân viên.

## Kiến trúc

```text
Browser camera
    -> user API
    -> YOLO face detector
    -> liveness classifier
    -> FaceNet 512D embedding
    -> pgvector cosine search
    -> attendance service
    -> PostgreSQL
```

Các lớp được tách theo trách nhiệm:

- `web/`: Flask app factory, authentication, CSRF và routes.
- `modules/repositories/`: chỉ xử lý truy vấn PostgreSQL.
- `modules/*_service.py`: business rules và use cases.
- `modules/detector.py`, `modules/liveness.py`, `modules/recognition.py`: adapter cho model và matching.
- `templates/`, `static/`: giao diện admin/user.

## Cài đặt

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
```

Tạo `.env` từ `.env.example` và thay toàn bộ placeholder. Không commit `.env` hoặc credentials thật.
Trong production đặt `APP_ENV=production`, dùng `SECRET_KEY` ngẫu nhiên tối thiểu 32 ký tự,
`ADMIN_PASSWORD_HASH`, `DB_SSLMODE=verify-full` và `RATELIMIT_STORAGE_URI` trỏ tới Redis dùng chung.

PostgreSQL cần cài extension `pgvector`, sau đó khởi tạo schema:

```bash
psql -U postgres -d face_db -f init_db.sql
```

Nếu database đã được tạo bằng phiên bản cũ, chạy thêm:

```bash
psql -U postgres -d face_db -f migrations/001_harden_existing_schema.sql
```

Mặc định hệ thống dùng Haar face detector để không phụ thuộc vào checkpoint YOLO không hợp lệ. Khi có model YOLO đã fine-tune cho khuôn mặt, đặt `DETECTOR_BACKEND=yolo` và trỏ `YOLO_MODEL_PATH` tới checkpoint đó. `models/yolov8m.pt` trong repository là checkpoint COCO tổng quát, không được dùng để phát hiện khuôn mặt.

## Chạy ứng dụng

Mở hai terminal riêng:

```bash
python app_admin.py
python app_user.py
```

- Admin: `http://localhost:5000/admin/login`
- User check-in/out: `http://localhost:8000/`

Đăng nhập admin bằng `ADMIN_USERNAME` và mật khẩu tương ứng với `ADMIN_PASSWORD_HASH` trong `.env`.
`ADMIN_PASSWORD` chỉ còn là fallback cho development.

## Luồng tính năng

1. Admin thêm nhân viên bằng file ảnh hoặc webcam.
2. Hệ thống yêu cầu đúng một khuôn mặt, tạo embedding và lưu cùng employee ID trong một transaction.
3. User gửi ảnh camera tới `/api/recognize`.
4. Hệ thống kiểm tra một khuôn mặt, liveness và nhận diện.
5. Attendance service kiểm tra loại chấm công, trạng thái đúng giờ và chống ghi trùng trong khoảng thời gian cấu hình.

## Lưu ý vận hành

- Chạy `main_recognition.py` hoặc `python -m modules.add_face` chỉ khi máy có camera GUI và cài `opencv-python`.
- `DEVICE=auto` tự chọn CUDA nếu khả dụng, ngược lại dùng CPU.
- Một nhân viên có thể có nhiều embedding trong tương lai; schema hiện tại đã giữ `employee_id` để hỗ trợ mở rộng.
- API nhận diện được giới hạn theo IP; production phải dùng Redis để giới hạn nhất quán giữa nhiều worker.
- API yêu cầu session token được cấp từ trang kiosk; đây là lớp chống request giả mạo, không thay thế firewall/mTLS khi mở mạng.
- Production bắt buộc khai báo `LIVENESS_MODEL_SHA256`; nếu dùng YOLO phải khai báo thêm `YOLO_MODEL_SHA256`.
- Không dùng `app.run` trực tiếp trên Internet. Đặt reverse proxy HTTPS, firewall và DB role riêng cho ứng dụng.

## Tạo cấu hình bảo mật

Tạo hash mật khẩu admin:

```bash
python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('thay-mat-khau-ngay'))"
```

Tạo SHA-256 cho model:

```bash
python -c "from pathlib import Path; import hashlib; p=Path('models/best_model.pth'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

Credential DB cũ đã từng xuất hiện trong lịch sử Git của dự án. Phải đổi credential trước khi
deploy và làm theo [SECURITY.md](SECURITY.md) để kiểm tra/purge lịch sử sau khi thống nhất với team.

## Kiểm tra code

```bash
python -m unittest discover -s tests -v
ruff check .
```
