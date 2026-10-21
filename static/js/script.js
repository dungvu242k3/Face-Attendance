const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const checkinBtn = document.getElementById('checkin-btn');
const checkoutBtn = document.getElementById('checkout-btn');
const employeeName = document.getElementById('employee-name');
const statusEl = document.getElementById('status');
const apiToken = document.querySelector('meta[name="user-api-token"]')?.content || '';

let cameraStream = null;
let requestInProgress = false;

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error('Trình duyệt không hỗ trợ camera');
  }
  cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = cameraStream;
  await video.play();
}

startCamera().catch((error) => {
  console.error('Không thể mở camera:', error);
  statusEl.textContent = 'Lỗi camera: ' + error.message;
});

function captureImage() {
  if (!video.videoWidth || !video.videoHeight) {
    throw new Error('Camera chưa sẵn sàng');
  }
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.8);
}

async function sendForRecognition(imageData, type) {
  const response = await fetch('/api/recognize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-User-Session-Token': apiToken,
    },
    body: JSON.stringify({ image: imageData, type })
  });
  const data = await response.json();
  if (!response.ok && data.status === 'error') {
    throw new Error(data.message || 'Không thể xử lý ảnh');
  }
  return data;
}

function showResult(result) {
  employeeName.textContent = result.name || 'Không xác định';
  statusEl.textContent = result.message || 'Không nhận diện được';
}

async function handleAttendance(type) {
  if (requestInProgress) return;
  requestInProgress = true;
  checkinBtn.disabled = true;
  checkoutBtn.disabled = true;
  statusEl.textContent = `Đang xử lý ${type}...`;

  try {
    showResult(await sendForRecognition(captureImage(), type));
  } catch (error) {
    employeeName.textContent = 'Không xác định';
    statusEl.textContent = error.message || 'Không thể kết nối đến máy chủ';
  } finally {
    requestInProgress = false;
    checkinBtn.disabled = false;
    checkoutBtn.disabled = false;
  }
}

checkinBtn.addEventListener('click', () => handleAttendance('checkin'));
checkoutBtn.addEventListener('click', () => handleAttendance('checkout'));

window.addEventListener('beforeunload', () => {
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
});
