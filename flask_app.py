from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import re
# import router
from router.main import main_bp

class ServerApp():
    def __init__(self, _HOST_IP, _HOST_PORT, _IS_DEBUG):
        self.HOST_IP = _HOST_IP
        self.HOST_PORT = _HOST_PORT
        self.IS_DEBUG = _IS_DEBUG
        
        self.app = Flask(__name__)
        CORS(self.app)
        
        # regi blueprints
        self.app.register_blueprint(main_bp)
            
    def get_app(self):
        return self.app
    
    def run(self):
        self.app.run(self.HOST_IP, self.HOST_PORT, self.IS_DEBUG)