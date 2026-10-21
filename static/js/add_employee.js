let cameraStream = null;

function previewFace(event) {
  const file = event.target.files[0];
  if (!file) return;
  document.getElementById('faceImageWebcam').value = '';
  const reader = new FileReader();
  reader.onload = () => {
    document.getElementById('previewWrap').innerHTML = `
      <img src="${reader.result}" class="w-32 h-32 object-cover rounded-lg mx-auto mb-3 border">
      <p class="text-green-600 text-sm">Ảnh đã chọn</p>`;
  };
  reader.readAsDataURL(file);
}

async function openCamera() {
  const wrap = document.getElementById('cameraWrap');
  try {
    wrap.classList.remove('hidden');
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
    document.getElementById('camera').srcObject = cameraStream;
  } catch (error) {
    wrap.classList.add('hidden');
    alert(`Không thể mở camera: ${error.message}`);
  }
}

function closeCamera() {
  document.getElementById('cameraWrap').classList.add('hidden');
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
}

function capturePhoto() {
  const video = document.getElementById('camera');
  if (!video.videoWidth || !video.videoHeight) {
    alert('Camera chưa sẵn sàng');
    return;
  }
  const canvas = document.getElementById('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataUrl = canvas.toDataURL('image/png');
  document.getElementById('faceFile').value = '';
  document.getElementById('previewWrap').innerHTML = `
    <img src="${dataUrl}" class="w-32 h-32 object-cover rounded-lg mx-auto mb-3 border">
    <p class="text-green-600 text-sm">Ảnh đã chụp</p>`;
  document.getElementById('faceImageWebcam').value = dataUrl;
  closeCamera();
}

function resizeImage(base64Str, maxWidth = 800, maxHeight = 800) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement('canvas');
      const ratio = Math.min(1, maxWidth / image.width, maxHeight / image.height);
      canvas.width = image.width * ratio;
      canvas.height = image.height * ratio;
      canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL('image/jpeg', 0.8));
    };
    image.onerror = () => reject(new Error('Không thể đọc ảnh webcam'));
    image.src = base64Str;
  });
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('faceFile').addEventListener('change', previewFace);
  document.getElementById('chooseFaceBtn').addEventListener('click', () => {
    document.getElementById('faceFile').click();
  });
  document.getElementById('openCameraBtn').addEventListener('click', openCamera);
  document.getElementById('capturePhotoBtn').addEventListener('click', capturePhoto);
  document.getElementById('closeCameraBtn').addEventListener('click', closeCamera);

  document.getElementById('formAddEmployee').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const webcamInput = form.querySelector('#faceImageWebcam');

    try {
      if (webcamInput.value) {
        formData.set('face_image_webcam', await resizeImage(webcamInput.value));
      }
      const response = await fetch('/admin/add_employee', {
        method: 'POST',
        body: formData,
        headers: { 'X-CSRF-Token': getCsrfToken() }
      });
      const data = await response.json();
      alert(data.message || 'Xong');
      if (response.ok && data.status === 'ok') window.location.href = '/admin/employees';
    } catch (error) {
      alert(error.message || 'Không thể kết nối đến máy chủ');
    }
  });
});

window.addEventListener('beforeunload', closeCamera);
