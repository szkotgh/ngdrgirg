import base64
import os
import re
from flask import Blueprint, render_template, jsonify, request
from dotenv import load_dotenv
from router.face.main import face_bp

load_dotenv()

PROCESS_URL = os.environ['PROCESS_URL']

main_bp = Blueprint('main_bp', __name__, url_prefix='/')
main_bp.register_blueprint(face_bp)

@main_bp.route('/', methods=['GET'])
def home():
    print(request.url)
    return render_template('index.html', prcs_url=PROCESS_URL)