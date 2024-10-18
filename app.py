import os
from dotenv import load_dotenv
# get flask app
from flask_app import ServerApp

load_dotenv()
HOST_IP = os.environ['HOST_IP']
HOST_PORT = os.environ['HOST_PORT']
PROCESS_URL = os.environ['PROCESS_URL']

server_app = ServerApp(HOST_IP, HOST_PORT, True)
server_app.run()