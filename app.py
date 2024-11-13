import os
import requests
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import hashlib

start_index = 1
end_index = 512

url = 'https://search.i815.or.kr/dictionary/moreSearchResult.do'
data_list = []

if not os.path.exists('faces'):
    os.makedirs('faces')

for i in tqdm(range(start_index, end_index + 1)):
    params = {
        "index": i,
        "searchWord": "",
        "searchType": "all",
    }
    
    response = requests.get(url, params=params)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    for person in soup.select('li.thumb_txt_case01'):
        name = person.select_one('strong.dict_txt_title').text.strip()
        movement = person.select_one('li.line_1').text.split(':')[-1].strip()
        organization = person.select_one('li.line_2').text.split(':')[-1].strip()
        activities = person.select_one('li:nth-of-type(3)').text.split(':')[-1].strip()
        content = person.select_one('li.under_txt').text.split(':')[-1].strip()
        
        img_tag = person.select_one('div.thumb img')
        img_url = img_tag['src']
        if img_url == '/img/service/none_profile.png':
            img_path = None
        else:
            img_response = requests.get(img_url)
            img_hash = hashlib.md5(img_response.content).hexdigest()
            img_path = f'faces/{img_hash}.jpg'
            with open(img_path, 'wb') as img_file:
                img_file.write(img_response.content)
        
        data_list.append({
            'name': name,
            'movement': movement,
            'organization': organization,
            'activities': activities,
            'content': content,
            'image': img_path
        })

with open('info.json', 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)