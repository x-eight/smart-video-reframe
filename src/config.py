from dotenv import load_dotenv
import os

load_dotenv()

config = {}

# Define model directory relative to the project root or absolute path
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
