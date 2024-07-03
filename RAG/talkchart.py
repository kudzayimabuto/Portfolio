import streamlit as st
import ollama
import chromadb
from PyPDF2 import PdfReader
from io import BytesIO
import pandas as pd
import hashlib
import re
import plotly.graph_objects as go

def ingest_pdf(file_bytes):
    pdf_reader = PdfReader(BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def ingest_csv(file_bytes):
    csv_data = pd.read_csv(BytesIO(file_bytes))
    return csv_data.to_string(index=False), csv_data

def get_or_create_collection(client, document_name):
    collection_name = hashlib.md5(document_name.encode()).hexdigest()
    collections = client.list_collections()
    if collection_name in [collection.name for collection in collections]:
        collection = client.get_collection(name=collection_name)
    else:
        collection = client.create_collection(name=collection_name)
    return collection

def add_documents_to_collection(collection, documents):
    for i, document in enumerate(documents):
        response = ollama.embeddings(model="mxbai-embed-large", prompt=document)
        embedding = response["embedding"]
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[document]
        )

def generate_plot(df, plot_code):
    local_vars = {'df': df}
    exec(plot_code, {}, local_vars)
    return local_vars['fig']

st.set_page_config(layout="wide")
st.title("Talk Chart")

uploaded_files = st.file_uploader("Upload PDF or CSV files", type=["pdf", "csv"], accept_multiple_files=True)

llm_model = st.selectbox("Select LLM Model", ["codegemma", "llama3", "mistral", "deepseek-coder:6.7b"])

client = chromadb.Client()

if 'history' not in st.session_state:
    st.session_state.history = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        document_name = uploaded_file.name
        documents = []

        if uploaded_file.type == "application/pdf":
            text = ingest_pdf(file_bytes)
            documents.append(text)
        elif uploaded_file.type == "text/csv":
            text, csv_data = ingest_csv(file_bytes)
            documents.append(text)
            st.write("CSV Data:")
            st.write(csv_data)

        collection = get_or_create_collection(client, document_name)

        add_documents_to_collection(collection, documents)
        st.success(f"Document '{document_name}' ingested and embeddings created!")

        if 'collection_name' not in st.session_state:
            st.session_state.collection_name = collection.name
        else:
            st.session_state.collection_name = collection.name

prompt = st.text_input("Enter your query:")
if st.button("Submit") and 'collection_name' in st.session_state:
    collection = client.get_collection(name=st.session_state.collection_name)

    if uploaded_file.type == "text/csv":
        system_prompt = (
            "Generate Python code that analyzes this data using pandas and plots using Plotly for visualizations of the data. "
            "In the Python code ingest using data = df.copy() and do not use pd.read_csv()."
            "Do not display the plot or Comment out fig.show() it is not required. "
            "Only use data = df.copy() do not generate your own data"
            "Here is the data:\n\n")
        full_prompt = f"{system_prompt}\n\n{csv_data.head(5).to_string(index=False)}\n\nPrompt: {prompt}"

        response = ollama.generate(
            model=llm_model,
            prompt=full_prompt
        )
        generated_code = response['response']


        code_match = re.search(r'```python\n(.*?)\n```', generated_code, re.DOTALL)
        if code_match:
            plot_code = code_match.group(1)

            try:
                fig = generate_plot(csv_data, plot_code)
                fig.update_layout(width=1000, height=600)
                st.plotly_chart(fig, use_container_width=True)


                st.session_state.history.append({
                    'prompt': prompt,
                    'code': plot_code,
                    'chart': fig
                })
            except Exception as e:
                st.error(f"Error generating plot: {e}")
                st.experimental_rerun()

    else:
        response = ollama.embeddings(
            prompt=prompt,
            model="mxbai-embed-large"
        )
        results = collection.query(
            query_embeddings=[response["embedding"]],
            n_results=1
        )
        data = results['documents'][0][0]

        output = ollama.generate(
            model=llm_model,
            prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
        )
        st.write(f"AI: {output['response']}")

st.header("History")
for idx, entry in enumerate(st.session_state.history):
    st.subheader(f"Entry {idx+1}")
    st.write(f"Prompt: {entry['prompt']}")
    st.code(entry['code'], language='python')
    st.plotly_chart(entry['chart'], use_container_width=True)
