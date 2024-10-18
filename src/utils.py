import json
import time
import os
import hashlib
import cv2
import numpy as np
from openvino.runtime import Core

def get_now_ftime(_format="%Y%m%d_%H%M%S"):
    return time.strftime(_format, time.localtime())

def convert_str_to_time(_str_time, _format="%Y%m%d_%H%M%S"):
    return time.strptime(_str_time, _format)

def gen_rhash(_length=64):
    random_bytes = os.urandom(32)
    hash_object = hashlib.sha256(random_bytes)
    random_hash = hash_object.hexdigest()
    return random_hash[:_length]

def get_option(_path : str | None = 'src/option.json'):
    if os.path.exists(_path) == False:
        return 1

    option_file = None
    with open(_path, 'r', encoding='utf-8') as f:
        option_file = json.load(f)
    
    return option_file
    
def get_file_ext(_file_name):
    if '.' not in _file_name:
        return None
    return _file_name.split('.')[-1] 

class FaceRecognition:
    def __init__(self, _detection_model_path: str, _reid_model_path: str, _device: str = "CPU"):
        self.core = Core()
        self.detection_model = self.load_model(_detection_model_path, _device)
        self.reid_model = self.load_model(_reid_model_path, _device)
        self.detection_input_shape = self.detection_model.inputs[0].shape
        self.reid_input_shape = self.reid_model.inputs[0].shape

    def load_model(self, _model_path: str, _device: str):
        model = self.core.read_model(f"{_model_path}.xml", f"{_model_path}.bin")
        compiled_model = self.core.compile_model(model, _device)
        return compiled_model

    def preprocess_image(self, _image_path):
        image = cv2.imread(_image_path)
        image = cv2.resize(image, (self.detection_input_shape[3], self.detection_input_shape[2]))
        
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = image.reshape(1, *image.shape)
        return image

    def extract_face(self, _image, _detection_threshold=0.5):
        detections = self.detection_model.infer_new_request({0: _image})[self.detection_model.outputs[0]]
        
        for detection in detections[0][0]:
            confidence = detection[2]
            if confidence > _detection_threshold:
                xmin = int(detection[3] * _image.shape[3])
                ymin = int(detection[4] * _image.shape[2])
                xmax = int(detection[5] * _image.shape[3])
                ymax = int(detection[6] * _image.shape[2])
                face = _image[0, :, ymin:ymax, xmin:xmax]

                if face is None or face.size == 0:
                    continue
                
                face = cv2.resize(face.transpose(1, 2, 0), (self.reid_input_shape[3], self.reid_input_shape[2]))
                return face.transpose(2, 0, 1).reshape(1, 3, self.reid_input_shape[2], self.reid_input_shape[3])
        
        return None

    def calculate_similarity(self, _base_face_embedding, _input_face_embedding):
        base_norm = np.linalg.norm(_base_face_embedding)
        input_norm = np.linalg.norm(_input_face_embedding)
        similarity = np.dot(_base_face_embedding, _input_face_embedding) / (base_norm * input_norm)
        return similarity

    def compare_faces(self, _base_face_path, _input_face_path):
        # 전처리
        base_image = self.preprocess_image(_base_face_path)
        input_image = self.preprocess_image(_input_face_path)
        
        # 얼굴 추출
        base_face = self.extract_face(base_image)
        input_face = self.extract_face(input_image)

        if base_face is None or input_face is None:
            return "Could not detect one or both faces."

        # 얼굴 임베딩 추출
        base_face_embedding = self.reid_model.infer_new_request({0: base_face})[self.reid_model.outputs[0]]
        input_face_embedding = self.reid_model.infer_new_request({0: input_face})[self.reid_model.outputs[0]]

        # 유사도 계산
        similarity = self.calculate_similarity(base_face_embedding.flatten(), input_face_embedding.flatten())
        return similarity

    def is_face(self, _image_path, _detection_threshold=0.5):
        image = self.preprocess_image(_image_path)
        face = self.extract_face(image, _detection_threshold)
        return face is not None
