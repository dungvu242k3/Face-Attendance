const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const checkinBtn = document.getElementById('checkin-btn');
const checkoutBtn = document.getElementById('checkout-btn');
const employeeName = document.getElementById('employee-name');
const statusEl = document.getElementById('status');

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(err => {
    console.error("Không thể mở camera:", err);
    statusEl.textContent = "Lỗi camera: " + err.message;
  });

function captureImage() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg');
}

async function sendForRecognition(imageData, type) {
  const resp = await fetch('/api/recognize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageData, type: type })
  });
  return await resp.json();
}

function showResult(result) {
  if (result.status === "ok") {
    employeeName.textContent = result.name;
    statusEl.textContent = result.message;
  } else {
    employeeName.textContent = result.name || "Không xác định";
    statusEl.textContent = result.message || "Không nhận diện được";
  }
}

checkinBtn.addEventListener('click', async () => {
  statusEl.textContent = "Đang xử lý Checkin...";
  const imgData = captureImage();
  const result = await sendForRecognition(imgData, 'Checkin');
  showResult(result);
});

checkoutBtn.addEventListener('click', async () => {
  statusEl.textContent = "Đang xử lý Checkout...";
  const imgData = captureImage();
  const result = await sendForRecognition(imgData, 'Checkout');
  showResult(result);
});

async function sendForRecognition(imageData, type) {
  const resp = await fetch('/api/recognize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageData, type: type })
  });
  return await resp.json();
}
