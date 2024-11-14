import os
import numpy as np
from tqdm import tqdm
import json
from src.FaceRecognition import FaceRecognition
import src.utils as utils

AI = FaceRecognition('./src/models/face-detection-adas-0001', './src/models/face-reidentification-retail-0095', 'CPU')

start_index = 1
end_index = 512

INFO_JSON_PATH = './info.json'
FACES_DIR_PATH = './faces'
PAGES_DIR_PATH = './pages'
EBD_DIR_PATH = './embeddings'

if not os.path.exists(EBD_DIR_PATH):
    os.makedirs(EBD_DIR_PATH)

with open(INFO_JSON_PATH, 'r', encoding='utf-8') as f:
    infos = json.load(f)
    
for i in tqdm(range(start_index, end_index + 1)):
    info = infos[i]
    
    if info['image'] is None:
        continue
    
    image_path = os.path.join(FACES_DIR_PATH, info['image'])
    embedding = None
    
    try:
        face_embedding = AI.get_face_embedding(image_path)
        embedding_filename = f'{utils.gen_hash(image_path)}.npy'
        embedding_path = os.path.join(EBD_DIR_PATH, embedding_filename)
        embedding = embedding_filename
        
        np.save(embedding_path, face_embedding)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        embedding = None
    
    infos[i]['embedding'] = embedding
    
with open('./update_info.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(infos, ensure_ascii=False, indent=4))