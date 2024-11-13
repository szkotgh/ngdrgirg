import re

def clean_text(text):
    cleaned_text = ''.join(text.split())
    return cleaned_text

def extract_id(onclick_text):
    match = re.search(r'\d+', onclick_text)
    if match:
        return match.group(0)
    return None