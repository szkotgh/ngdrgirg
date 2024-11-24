import base64
import json
import re
import os
from flask import Blueprint, redirect, render_template, jsonify, request, send_file
import numpy as np
import src.utils as utils

OPTION = utils.get_option()

army_bp = Blueprint('army', __name__, url_prefix='/army')
Face_Detector = utils.FaceRecognition(OPTION['dect_model_path'], OPTION['reid_model_path'])

@army_bp.route('/compare', methods=['POST'])
def compare():
    TEMP_DIR = 'src/temp'
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    try:
        data = request.get_json()
        image_data = data['image']
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        image_bytes = base64.b64decode(image_data)

        image_id = utils.gen_rhash()
        image_path = os.path.join(TEMP_DIR, f'{image_id}.jpg')
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        image_embedding = Face_Detector.get_face_embedding(image_path)
        if image_embedding is None:
            return jsonify({'status': 'error', 'status_msg': 'No face detected in the image'})
        
        embeddings_path = os.path.join('db', 'army_db', 'embeddings')
        embeddings = os.listdir(embeddings_path)
        similarities = []
        for embedding in embeddings:
            target_embedding = np.load(os.path.join(embeddings_path, embedding))
            distance = Face_Detector.calculate_similarity(image_embedding, target_embedding)
            similarities.append([embedding, distance])
        similarities = sorted(similarities, key=lambda x: x[1])
        top_similarities = similarities[:5]
            
        with open(os.path.join('db', 'army_db', 'info.json'), 'r') as f:
            army_infos = json.load(f)
             
        top_army_infos = []
        for similarity in top_similarities:
            for army_info in army_infos:
                if similarity[0] == army_info.get('embedding'):
                    army_info['distance'] = round(abs(similarity[1])*100, 2)
                    top_army_infos.append(army_info)
                    break
        
        # 임시 결과를 저장하고 리다이렉트
        result_id = utils.gen_rhash()
        result_path = os.path.join(TEMP_DIR, f'{result_id}.json')
        result_json = {
            'image_id': image_id,
            'top_army_infos': top_army_infos
        }
        with open(result_path, 'w') as f:
            json.dump(result_json, f)
        
        return jsonify({'status': 'success', 'rstId': result_id})
    except Exception as e:
        return jsonify({'status': 'error', 'status_msg': f"An unknown error has occurred: {str(e)}"})

@army_bp.route('/compare/result/<result_id>', methods=['GET'])
def result(result_id):
    result_path = os.path.join('src/temp', f'{result_id}.json')
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            army_infos = json.load(f)
        return render_template('/army/compare.html', army_infos=army_infos['top_army_infos'], yourface_id=army_infos['image_id'])
    else:
        return redirect('/')

@army_bp.route('/view', methods=['GET'])
def view():
    return render_template('/army/view.html')

@army_bp.route('/api/army_infos', methods=['GET'])
def get_army_infos():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 36))
    query = request.args.get('query', '').lower()
    
    with open('./db/army_db/info.json', 'r') as f:
        army_infos = json.load(f)
    
    if query:
        army_infos = [info for info in army_infos if query in info['name'].lower() or query in info['content'].lower() or query in info['activities'].lower()]
    
    total = len(army_infos)
    start = (page - 1) * per_page
    end = start + per_page
    data = army_infos[start:end]
    
    return jsonify({
        'data': data,
        'total': total,
        'page': page,
        'per_page': per_page
    })