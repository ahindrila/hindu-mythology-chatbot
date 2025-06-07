import streamlit as st
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import google.generativeai as genai
from google.colab import userdata # Note: userdata will not work directly in a deployed Streamlit app
import os

# Function to load data and create chunks
@st.cache_resource # Cache the data loading and chunking
def load_and_chunk_data(file_path):
    df = pd.read_csv(file_path)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = []
    for index, row in df.iterrows():
        # Handle potential None values in 'text' column
        page_text = row['text'] if pd.notna(row['text']) else ""
        page_title = row['Unnamed: 0']

        text_with_title = f"Title: {page_title}\n\n{page_text}"
        page_chunks = text_splitter.create_documents([text_with_title])

        for chunk in page_chunks:
            chunk.metadata = {"source": page_title}
        chunks.extend(page_chunks)
    return chunks

# Function to setup RAG components
@st.cache_resource # Cache the model and vector store setup
def setup_rag(chunks):
    # Initialize the embedding model
    # Consider using a different embedding approach for deployment if needed
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize the vector store
    vectorstore = Chroma.from_documents(chunks, embedding_model)
    return vectorstore

# Streamlit App Title
st.title("Hindu Mythology Chatbot (RAG)")

# Load data and setup RAG
data_file_path = 'hindu_mythology_data.csv' # Make sure this file is available in your deployment environment
if not os.path.exists(data_file_path):
    st.error(f"Data file not found at {data_file_path}. Please upload the data file.")
else:
    chunks = load_and_chunk_data(data_file_path)
    vectorstore = setup_rag(chunks)

    # --- Chatbot Logic ---
    st.write("Ask me anything about Hindu Mythology!")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Enter your question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Retrieve relevant documents from the vector store
            retrieved_docs = vectorstore.similarity_search(prompt, k=3) # Adjust k as needed

            # Use the generative model to answer the question
            # IMPORTANT: Replace userdata.get('GOOGLE_API_KEY') with a secure way to access your API key in deployment
            # Streamlit Cloud uses Streamlit Secrets for this: https://docs.streamlit.io/deploy/streamlit-cloud/secrets-management
            api_key = os.environ.get("GOOGLE_API_KEY") or userdata.get('GOOGLE_API_KEY') # Attempt to get from env or secrets
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    llm = genai.GenerativeModel('gemini-1.5-flash-latest')

                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                    rag_prompt = f"""
                    You are a helpful chatbot that provides information about Hindu mythology based on the provided context.
                    Answer the following question based *only* on the context provided. If the context does not contain
                    enough information to answer the question, say "I don't have enough information to answer that question."

                    Context:
                    {context}

                    Question:
                    {prompt}

                    Answer:
                    """
                    response = llm.generate_content(rag_prompt)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})

                except Exception as e:
                    st.error(f"An error occurred while generating the response: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error while trying to generate a response."})

            else:
                st.warning("API key not found. Please ensure GOOGLE_API_KEY is set up in Streamlit Secrets.")
                st.session_state.messages.append({"role": "assistant", "content": "API key not found. Please ensure GOOGLE_API_KEY is set up in Streamlit Secrets."})
