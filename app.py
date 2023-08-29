import os
import PyPDF2
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager

if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'  ## 'collapsed' or 'expanded'

st.set_page_config(page_title="Chat PDF", page_icon=':shark:', initial_sidebar_state=st.session_state.sidebar_state)

@st.cache_data
def load_docs(files):
    st.info("`Loading document...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        else:
            st.warning('Please provide a valid .pdf document.', icon="⚠️")
    return all_text

@st.cache_resource
def create_retriever(_embeddings, splits):
    try:
        vectorstore = FAISS.from_texts(splits, _embeddings)
    except (IndexError, ValueError) as e:
        st.error(f"Error creating vectorstore: {e}")
        return
    return vectorstore.as_retriever(k=5)

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):
    st.info("`Processing document...`")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to process document")
        st.stop()
    return splits

def main():
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        header .decoration {
            display: none;
        }
        footer {visibility: hidden;}
        .css-card {
            border-radius: 0px;
            padding: 30px 10px 10px 10px;
            background-color: #f8f9fa;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 10px;
            font-family: "IBM Plex Sans", sans-serif;
        }
        .card-tag {
            border-radius: 0px;
            padding: 1px 5px 1px 5px;
            margin-bottom: 10px;
            position: absolute;
            left: 0px;
            top: 0px;
            font-size: 0.6rem;
            font-family: "IBM Plex Sans", sans-serif;
            color: white;
            background-color: green;
        }
        .css-zt5igj {left:0;}
        span.css-10trblm {margin-left:0;}
        div.css-1kyxreq {margin-top: -40px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.write(
    f"""
    <div style="display: flex; align-items: center; margin-left: 0;">
        <h1 style="display: inline-block;">Chat PDF</h1>
        <sup style="margin-left:5px;font-size:small; color: green;">beta</sup>
    </div>
    """,
    unsafe_allow_html=True,
        )
    
    st.sidebar.title("Menu")

    if 'openai_api_key' not in st.session_state:
        openai_api_key = st.text_input(
            'Please enter your OpenAI API key', value="", placeholder="Your OpenAI API key begins with sk-")
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            return
    else:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

    available_embeddings = [
        'OpenAIEmbeddings', 
        'HuggingFaceEmbeddings',
        'TensorflowHubEmbeddings',
        'SpacyEmbeddings'
    ]

    selected_embedding = st.sidebar.selectbox('Select Embedding:', available_embeddings)
    if selected_embedding == 'OpenAIEmbeddings':
        embeddings = OpenAIEmbeddings()
    elif selected_embedding == 'HuggingFaceEmbeddings':
        embeddings = HuggingFaceEmbeddings()
    elif selected_embedding == 'TensorflowHubEmbeddings':
        embeddings = TensorflowHubEmbeddings()
    elif selected_embedding == 'SpacyEmbeddings':
        embeddings = SpacyEmbeddings()

    uploaded_files = st.file_uploader("Upload a PDF document", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files

        loaded_text = load_docs(uploaded_files)
        st.write("Document loaded.")

        splits = split_texts(loaded_text, chunk_size=1000, overlap=0, split_method="RecursiveCharacterTextSplitter")
        st.write("Document processed.")

        embeddings = OpenAIEmbeddings()
        retriever = create_retriever(embeddings, splits)

        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = AsyncCallbackManager([callback_handler])

        chat_openai = ChatOpenAI(
            streaming=True, callback_manager=callback_manager, verbose=True, temperature=0)
        qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)

        user_question = st.text_input("", value="", placeholder="How can I help you?")
        if user_question:
            with st.spinner('Generating answer...'):
                answer = qa.run(user_question)
            st.write(answer)

if __name__ == "__main__":
    main()
