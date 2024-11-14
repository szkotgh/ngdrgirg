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
    
    if info['image'] == None:
        continue
    
    image_path = os.path.join(FACES_DIR_PATH, info['image'])
    embedding = None
    
    try:
        emage_embedding = AI.get_face_embedding(image_path)
        embedding = f'{utils.gen_hash()}.npy'
        with open(os.path.join(EBD_DIR_PATH, embedding), 'wb') as f:
            np.save(f, embedding)
    except:
        pass
    
    infos[i]['embedding'] = embedding