import cv2
from camera import Camera
from detector import FaceDetector

from database import FaceDatabase

db = FaceDatabase(password="sat24042003") 
detector = FaceDetector(device="cuda")
cam = Camera()

name = input("Nhập tên: ")

while True:
    ret, frame = cam.read_frame()
    if not ret:
        break

    boxes = detector.detect_faces(frame)
    if boxes and len(boxes) > 0:
        embeddings = detector.get_embeddings()
        if embeddings and len(embeddings) > 0:
            db.add_face(name, embeddings[0], filename="webcam_capture.jpg")
            print(f" Đã thêm: {name}")
            break

    cv2.imshow("Add Face", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
db.close()
