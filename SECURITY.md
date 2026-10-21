# Security runbook

## Trước khi deploy

1. Đổi mật khẩu DB cũ và kiểm tra log truy cập. Không tái sử dụng credential đã từng nằm
   trong Git history.
2. Đặt `APP_ENV=production` và cấu hình:
   - `SECRET_KEY` ngẫu nhiên tối thiểu 32 ký tự.
   - `ADMIN_PASSWORD_HASH`, không dùng `ADMIN_PASSWORD`.
   - `SESSION_COOKIE_SECURE=true`.
   - `DB_SSLMODE=verify-full` và `DB_SSLROOTCERT`.
   - `CORS_ORIGINS` chỉ gồm origin thật của kiosk.
   - `RATELIMIT_STORAGE_URI=redis://...` dùng backend chung cho mọi worker.
   - `LIVENESS_MODEL_SHA256` và `YOLO_MODEL_SHA256` nếu dùng YOLO.
3. Dùng DB role riêng, chỉ cấp quyền trên các bảng cần thiết; không chạy ứng dụng bằng
   role `postgres`.
4. Đặt reverse proxy HTTPS, firewall và giới hạn truy cập admin theo VPN/LAN.

## Dọn credential trong lịch sử Git

Đây là thao tác rewrite lịch sử, cần backup và thống nhất với mọi người đang dùng repository.

```bash
git clone --mirror <repository-url> Face-Attendance.git
cd Face-Attendance.git
git filter-repo --replace-text replacements.txt
git push --force --all origin
git push --force --tags origin
```

`replacements.txt` phải chứa secret cũ theo format của `git filter-repo`; không commit file này.
Sau khi rewrite, mọi clone cũ phải được xóa hoặc clone lại. Việc rewrite không thay thế cho
đổi mật khẩu DB.

## Kiểm tra trước release

```bash
pip-audit -r requirements.txt
bandit -r . -x ./.git,./models,./modules/__pycache__,./tests
python -m unittest discover -s tests -v
```

## Dependency exception cần xử lý riêng

`facenet-pytorch==2.6.0` hiện khóa `torch<2.3` và `Pillow<10.3`, nên không thể nâng
hai thư viện này chỉ bằng `requirements.txt` mà vẫn giữ resolver hợp lệ. Đây là lý do
`pip-audit` còn báo lỗi cho `torch 2.2.2` và `Pillow 10.2.0`. Trước khi mở dịch vụ ra
Internet, cần chuyển sang embedding runtime được duy trì hoặc fork/vendor adapter đã
được kiểm thử với Torch/Pillow bản vá. Trong thời gian chuyển đổi, model hash bắt buộc,
`torch.load(weights_only=True)` và việc không dùng Pillow trên HTTP image path chỉ là
biện pháp giảm rủi ro, không phải thay thế cho nâng dependency.
