# Talk Chart
Talk Chart is a Streamlit-based web application that allows users to upload PDF and CSV files, generate embeddings, and create visualizations using Plotly.

## Requirements

- Python 3.10
- pip (Python package installer)

## Installation

1. Download and install Ollama from their website https://ollama.com/download/mac
2. Open your terminal and run the following commands
   ollama pull codegemma
   ollama pull mxbai-embed-large
3. Using your terminal navigate to where you stored talkchart.py and create a python3.10 virtual machine(download python 3.10 from here: https://www.python.org/downloads/release/python-3100/)
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt

## Run talkchart
   streamlit run app.py
