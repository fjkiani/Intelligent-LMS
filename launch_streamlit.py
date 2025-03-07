import os
import sys
import subprocess

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the Streamlit app
streamlit_app = os.path.join(script_dir, "final_rag_sys", "streamlit_mock_rag.py")

# Run the Streamlit app
cmd = [sys.executable, "-m", "streamlit", "run", streamlit_app]
subprocess.run(cmd) 