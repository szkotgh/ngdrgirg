import os
import requests
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import hashlib
import utils

start_index = 1
end_index = 512

url = 'https://search.i815.or.kr/dictionary/moreSearchResult.do'
base_url = 'https://search.i815.or.kr'
data_list = []

INFO_JSON_PATH = './info.json'
FACES_DIR_PATH = './faces'
PAGES_DIR_PATH = './pages'

if not os.path.exists('faces'):
    os.makedirs('faces')
if not os.path.exists('pages'):
    os.makedirs('pages')

for i in tqdm(range(start_index, end_index + 1)):
    params = {
        "index": i,
        "searchWord": "",
        "searchType": "all",
    }
    
    response = requests.get(url, params=params)
    with open(os.path.join(PAGES_DIR_PATH, f'{i}.html'), 'w', encoding='utf-8') as f:
        f.write(response.text)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    for person in soup.select('li.thumb_txt_case01'):
        name = person.select_one('strong.dict_txt_title').text.strip()
        name = utils.clean_text(name)
        
        id = person.select_one('div.mouse_over')['onclick']
        id = utils.extract_id(id)
        
        workseries = person.select_one('li.line_1').text.split(':')[-1].strip()
        workseries = utils.clean_text(workseries)
        
        organization = person.select_one('li.line_2').text.split(':')[-1].strip()
        organization = utils.clean_text(organization)
        
        activities = person.select_one('li:nth-of-type(3)').text.split(':')[-1].strip()
        
        content = person.select_one('li.under_txt').text.split(':')[-1].strip()
        
        img_tag = person.select_one('div.thumb img')
        img_url = img_tag['src']
        if img_url == '/img/service/none_profile.png':
            img_path = None
        else:
            if img_url.startswith('/'):
                img_url = base_url + img_url
            img_response = requests.get(img_url)
            if img_response.status_code == 200:
                img_hash = hashlib.md5(img_response.content).hexdigest()
                img_name = f'{img_hash}.jpg'
                with open(os.path.join(FACES_DIR_PATH, img_name), 'wb') as img_file:
                    img_file.write(img_response.content)
            else:
                img_path = None
        
        data_list.append({
            'name': name,
            'id': id,
            'workseries': workseries,
            'organization': organization,
            'activities': activities,
            'content': content,
            'image': img_path
        })

    with open(INFO_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
