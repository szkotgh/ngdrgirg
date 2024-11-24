import os
import random
import json
from flask import Blueprint, render_template, jsonify, request, send_file
from dotenv import load_dotenv
from router.face.main import face_bp
from router.army.main import army_bp

load_dotenv()

PROCESS_URL = os.environ['PROCESS_URL']

main_bp = Blueprint('main', __name__, url_prefix='/')
main_bp.register_blueprint(face_bp)
main_bp.register_blueprint(army_bp)

@main_bp.route('/', methods=['GET'])
def home():
    return render_template('index.html', prcs_url=PROCESS_URL)

@main_bp.route('/favicon.ico', methods=['GET'])
def favicon():
    return send_file('static/favicon.ico')

@main_bp.route('/robots.txt', methods=['GET'])
def robots():
    return send_file('static/robots.txt')

@main_bp.route('/get_wise', methods=['GET'])
def get_wise():
    with open('db/wise.json', 'r') as f:
        wise = json.load(f)
    
    return jsonify({'status': 'success', 'wise': wise[random.randint(0, len(wise) - 1)]})