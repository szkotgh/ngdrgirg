import base64
import re
import os
from flask import Blueprint, render_template, jsonify, request, send_file
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

@face_bp.route('/army/<image_name>', methods=['GET'])
def face(image_name):
    image_path = f"db/army_db/faces/{image_name}.jpg"
    if os.path.exists(image_path):
        try:
            return send_file(image_path, mimetype='image/jpeg')
        except Exception as e:
            return jsonify({'status': 'error', 'status_msg': f"Error loading image: {str(e)}"}), 500
    else:
        return jsonify({'status': 'error', 'status_msg': 'Invalid id'}), 404
    
@face_bp.route('/your/<face_id>', methods=['GET'])
def yourface(face_id):
    image_path = f"src/temp/{face_id}.jpg"
    if os.path.exists(image_path):
        try:
            response = send_file(image_path, mimetype='image/jpeg')
            os.remove(image_path)
            return response
        except Exception as e:
            return jsonify({'status': 'error', 'status_msg': f"Error loading image: {str(e)}"}), 500
    else:
        return jsonify({'status': 'error', 'status_msg': 'Invalid id'}), 404