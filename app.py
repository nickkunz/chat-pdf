import os
import PyPDF2
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager

if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'  ## 'collapsed' or 'expanded'

st.set_page_config(
    page_title = "Chat PDF",
    page_icon = ':shark:',
    initial_sidebar_state = st.session_state.sidebar_state
)

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

def create_retriever(_embeddings, splits):
    try:
        vectorstore = FAISS.from_texts(splits, _embeddings)
    except (IndexError, ValueError) as e:
        st.error(f"Error creating vectorstore: {e}")
        return
    return vectorstore.as_retriever(k=5)

def split_texts(text, chunk_size, overlap):
    st.info("`Processing document...`")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = overlap
    )

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to process document")
        st.stop()
    return splits

PREDEFINED_QUESTIONS = [
    {
        "question": "What is the main idea of this document?",
        "prompt": "Summarize the main idea of this document."
    },
    {
        "question": "What are 3 key points discussed in this document?",
        "prompt": "List the key points discussed in this document."
    },
    {
        "question": "Can you provide a brief overview of the document?",
        "prompt": "Give me a brief overview of the document."
    }
]

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
            label = 'Please enter your OpenAI API key', 
            value = "", 
            placeholder = "Your OpenAI API key begins with sk-"
        )
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            return
    else:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

    available_embeddings = [
        'OpenAI', 
        'HuggingFace'
    ]

    selected_embedding = st.sidebar.selectbox('Select Embedding Method:', available_embeddings)
    if selected_embedding == 'OpenAI':
        embeddings = OpenAIEmbeddings()
    elif selected_embedding == 'HuggingFace':
        embeddings = HuggingFaceEmbeddings()

    available_models = [
        'GPT-3.5',
        'GPT-4'
    ]

    selected_model = st.sidebar.selectbox('Select Model Type:', available_models)
    
    uploaded_files = st.file_uploader("Upload a PDF document", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files

        if 'predefined_answers' not in st.session_state:
            st.session_state.predefined_answers = None

        loaded_text = load_docs(uploaded_files)
        st.write("Document loaded.")

        splits = split_texts(
            text = loaded_text,
            chunk_size = 1000,
            overlap = 0
        )
        st.write("Document processed.")

        retriever = create_retriever(embeddings, splits)

        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = AsyncCallbackManager([callback_handler])

        if selected_model == 'GPT-3.5':
            model = ChatOpenAI(
                model_name = "gpt-3.5-turbo",
                streaming = True,
                callback_manager = callback_manager,
                verbose = True,
                temperature = 0
            )
        elif selected_model == 'GPT-4':
            model = ChatOpenAI(
                model_name = "gpt-4",
                streaming = True,
                callback_manager = callback_manager,
                verbose = True,
                temperature = 0
            )
        
        qa = RetrievalQA.from_chain_type(
            llm = model,
            retriever = retriever,
            chain_type = "stuff",
            verbose = True
        )

        st.markdown("---")  ## horizontal line

        user_question = st.text_input(
            label = "Ask a Question",
            value = "",
            placeholder = "How can I help you?"
        )

        user_prompt = st.text_area(
            label="Context (Optional):",
            value="",
            placeholder="Please provide any context or specific instructions here..."
        )

        if user_question:
            with st.spinner('Generating answer...'):
                answer = qa.run(f"{user_prompt} {user_question}")
            st.write(answer)

        st.markdown("---")  ## horizontal line
        
        if st.session_state.predefined_answers is None:
            with st.spinner('Generating answers for predefined questions...'):
                st.session_state.predefined_answers = [
                    qa.run(f"{question['question']} {question['prompt']}") for question in PREDEFINED_QUESTIONS
                ]

        for i, (q, a) in enumerate(zip(PREDEFINED_QUESTIONS, st.session_state.predefined_answers)):
            # st.markdown(f"<h3>Response {i + 1}</h3>", unsafe_allow_html=True)
            st.write(f"**Question:** {q['question']}")
            st.write(f"**Context:** {q['prompt']}")
            st.write(f"**Answer:** {a}\n")
            st.write("---")

if __name__ == "__main__":
    main()
