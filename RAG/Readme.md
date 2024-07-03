# Talk Chart

Talk Chart is a Streamlit-based web application that allows users to upload PDF and CSV files, generate embeddings, and create visualizations using Plotly.

## Requirements

- Python 3.10
- pip (Python package installer)

## Installation

1. Clone the repository or download the source code.

    ```bash
    git clone https://github.com/your-repo/talk-chart.git
    cd talk-chart
    ```

2. Create a virtual environment.

    **On Linux/MacOS:**

    ```bash
    python3.10 -m venv venv
    source venv/bin/activate
    ```

    **On Windows:**

    ```bash
    python3.10 -m venv venv
    venv\Scripts\activate
    ```

    This will create and activate a virtual environment named `venv` in your project directory. Your command prompt or terminal should now show a prefix `(venv)` indicating the virtual environment is active.

3. Install the required Python packages.

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app.

    ```bash
    streamlit run app.py
    ```

5. Open your web browser and navigate to `http://localhost:8501`.

## Features

- Upload PDF or CSV files.
- Generate embeddings for the uploaded documents using Ollama.
- Analyze CSV data and generate Plotly visualizations.
- Store and query documents using ChromaDB.
- Maintain a history of prompts and responses.

## File Structure

- `app.py`: Main application file containing the Streamlit code.
- `requirements.txt`: List of required Python packages.
- `README.md`: Instructions for setting up and running the application.

## Dependencies

- **Streamlit**: A fast way to build and share data apps.
- **Ollama**: Library for generating embeddings.
- **ChromaDB**: Database for storing and querying document embeddings.
- **PyPDF2**: Library for reading PDF files.
- **Pandas**: Data analysis and manipulation tool.
- **Plotly**: Graphing library for creating interactive visualizations.

## License
