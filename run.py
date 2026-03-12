import traceback
import sys
import os

# Ensure we are working from the project root and can import from 'src'
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from train_anomaly import train

try:
    train()
except Exception as e:
    with open('err.txt', 'w', encoding='utf-8') as f:
        traceback.print_exc(file=f)
    print("Error saved to err.txt")
