import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/ui/streamlit_app.py"])
