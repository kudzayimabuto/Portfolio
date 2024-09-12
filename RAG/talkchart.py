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

def transform_table(df, table_code):
    local_vars = {'df': df.copy()}  # Copy the data to avoid overwriting the original
    
    # Try executing the LLM-generated code
    try:
        exec(table_code, {}, local_vars)
        
        # Check if 'df' was modified, otherwise raise an error
        if 'df' in local_vars and isinstance(local_vars['df'], pd.DataFrame):
            return local_vars['df']  # Return the transformed DataFrame
        else:
            raise ValueError("Transformed DataFrame not found or not valid.")
    except Exception as e:
        raise RuntimeError(f"Error executing the transformation code: {e}")


st.set_page_config(layout="wide")
st.title("Talk Chart")

uploaded_files = st.file_uploader("Upload PDF or CSV files", type=["pdf", "csv"], accept_multiple_files=True)

llm_model = st.selectbox("Select LLM Model", ["codegemma", "codellama", "llama3.1", "mistral", "gemma2"])

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

            # Display CSV data for reference
            st.write("CSV Data:")
            st.write(csv_data)

            # Create a description of the CSV's structure to use for embedding
            csv_description = (
                f"CSV file '{document_name}' has {csv_data.shape[0]} rows and {csv_data.shape[1]} columns.\n"
                f"Columns: {', '.join(csv_data.columns)}\n"
                f"Sample data:\n{csv_data.head(5).to_string(index=False)}"
            )

            # Add the description to the list of documents for embedding
            documents.append(csv_description)

        collection = get_or_create_collection(client, document_name)

        add_documents_to_collection(collection, documents)
        st.success(f"Document '{document_name}' ingested and embeddings created!")

        if 'collection_name' not in st.session_state:
            st.session_state.collection_name = collection.name
        else:
            st.session_state.collection_name = collection.name

prompt = st.text_input("Enter your query (if you require a plot or a table enter: '/plot' or '/table' then enter prompt)")
if st.button("Submit") and 'collection_name' in st.session_state:
    collection = client.get_collection(name=st.session_state.collection_name)

    if uploaded_file.type == "text/csv":
        if prompt.startswith("/plot"):
            # Handle plot generation
            system_prompt = ("""Generate Python code to analyze the provided data using pandas and create visualizations using Plotly. Adhere to these guidelines:

                    1. Start with: data = df.copy()
                    2. Do not use pd.read_csv() or generate any synthetic data.
                    3. Perform data analysis using pandas functions.
                    4. Create one or more Plotly visualizations based on the analysis.
                    5. Do not include fig.show() or any display commands.
                    6. Assign the final plot to a variable named 'fig'.
                    7. Add brief comments explaining key analysis steps or visualization choices.
                    8. Wrap your code in triple backticks with 'python' specified.

                    Your code should follow this structure:
                    ```python
                    import pandas as pd
                    import plotly.graph_objects as go
                    # Import any other necessary modules if neccesary

                    data = df.copy()

                    # Data analysis code here

                    # Visualization code here
                    fig = go.Figure(
                        # Plotly figure configuration
                    )

                    # Any additional figure customization
                    fig.update_layout(
                        title="Your Plot Title",
                        # Other layout parameters
                    )"""
            )
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

        elif prompt.startswith("/table"):
            # Handle table transformation using LLM
            system_prompt = ("""
                                Generate Python code to transform the provided CSV data using pandas. Follow these strict guidelines:

                                1. Start with: data = df.copy()
                                2. Perform transformations on the 'data' DataFrame.
                                3. Do not use pd.read_csv() or generate any synthetic data.
                                4. After transformations, save the result in result_df the save as 'output.csv' using: result_df.to_csv('output.csv', index=False)
                                5. Do not include any print statements or display commands.
                                6. Do not include any code to read the output file after saving.
                                7. Wrap your code in triple backticks with 'python' specified.

                                Your code should follow this structure:
                                ```python
                                data = df.copy()
                                # Your transformation code here
                                result_df.to_csv('output.csv', index=False)
                                """
            )
            full_prompt = f"{system_prompt}\n\n{csv_data.head(5).to_string(index=False)}\n\nPrompt: {prompt}"

            response = ollama.generate(
                model=llm_model,
                prompt=full_prompt
            )
            generated_code = response['response']

            code_match = re.search(r'```python\n(.*?)\n```', generated_code, re.DOTALL)
            if code_match:
                table_code = code_match.group(1)

                try:
                    # Use the transform_table function to apply the transformation
                    transform_table(csv_data, table_code)
                    transformed_data = pd.read_csv("output.csv")
                    st.write("Transformed Table:")
                    st.write(transformed_data)

                    st.session_state.history.append({
                        'prompt': prompt,
                        'code': table_code,
                        'table': transformed_data
                    })
                except Exception as e:
                    st.error(f"Error transforming table: {e}")

        else:
            # Pass the CSV data and prompt to the LLM for detailed analysis
            system_prompt = (
                """Analyze the following CSV data and provide a detailed, structured analysis based on the user's prompt. The data is already ingested in pandas format. Follow these guidelines for your analysis:

                1. Data Overview:
                - Describe the structure of the dataset (number of rows and columns).
                - List and briefly explain the meaning of each column.
                - Identify the data types of each column.

                2. Descriptive Statistics:
                - Provide summary statistics for numerical columns (mean, median, min, max, standard deviation).
                - For categorical columns, show the number of unique values and the most frequent categories.

                3. Data Quality Assessment:
                - Check for missing values and their distribution across columns.
                - Identify any potential outliers or anomalies.
                - Comment on data consistency and any potential issues.

                4. Key Insights:
                - Identify and describe the main trends or patterns in the data.
                - Highlight any significant correlations between variables.
                - Point out any surprising or noteworthy findings.

                5. Specific Analysis:
                - Address the user's specific prompt or question about the data.
                - Provide relevant calculations or data manipulations to answer the user's query.

                6. Recommendations:
                - Suggest potential next steps for further analysis or data processing.
                - Recommend visualizations that could provide additional insights.

                Present the results in a clear, organized, and structured format:\n\n"""
            )

            full_prompt = f"{system_prompt}\n\n{csv_data.to_string(index=False)}"

            # Handle non-CSV files (e.g., PDF)
            response = ollama.embeddings(
                prompt=full_prompt,
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



    else:
        # Handle non-CSV files (e.g., PDF)
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
    if 'analysis' in entry:
        st.write(f"Analysis: {entry['analysis']}")
    if 'code' in entry:
        st.code(entry['code'], language='python')
    if 'table' in entry:
        st.write("Transformed Table:")
        st.write(entry['table'])
    if 'chart' in entry:
        st.plotly_chart(entry['chart'], use_container_width=True)
