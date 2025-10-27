from models.face_recognition import FaceRecognition
from database.employee_db import EmployeeDB
import cv2

face_model = FaceRecognition("models/weights/buffalo_s/w600k_mbf.onnx")
emp_db = EmployeeDB()

img = cv2.imread("path/to/your_face.jpg")
emb = face_model.get_face_embedding(img)
emp_db.add_employee("E001", emb, emb)  
