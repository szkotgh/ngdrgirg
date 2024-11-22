import base64
import re
import os
from flask import Blueprint, render_template, jsonify, request
import src.utils as utils

OPTION = utils.get_option()

face_bp = Blueprint('face', __name__, url_prefix='/face')
Face_Detector = utils.FaceRecognition(OPTION['dect_model_path'], OPTION['reid_model_path'])

@face_bp.route('/is_face', methods=['POST'])
def is_face():
    TEMP_PATH = f"{OPTION['TEMP_DIR_PATH']}/upload.jpg"
    try:
        data = request.get_json()
        image_data = data['image']
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        image_bytes = base64.b64decode(image_data)

        with open(TEMP_PATH, "wb") as f:
            f.write(image_bytes)

        is_face = Face_Detector.is_face(TEMP_PATH, 0.5)
        
        os.remove(TEMP_PATH)
        
        return jsonify({'status': 'normal', 'is_face': is_face})
    except Exception as e:
        return jsonify({'status': 'error', 'status_msg': f"An unknown error has occurred: {str(e)}"})

