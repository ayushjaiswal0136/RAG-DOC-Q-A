# Langchain modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Other modules
import os
import uuid
import tempfile
import streamlit as st
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit setup
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Q&A System")
st.write("Upload a PDF document and ask questions about its content.")

# Sidebar for user settings
with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, step=50)
    num_chunks = st.slider("Number of Chunks to Retrieve", 1, 10, 4, step=1)

    if not OPENAI_API_KEY:
        st.error("Please set your OpenAI API key in the .env file")
        st.stop()

# Load LLM
try:
    llm = ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0.7,
        max_tokens=1000
    )
except Exception as e:
    st.error(f"Error initializing LLM: {str(e)}")
    st.stop()

# Embedding function
def get_embedding_function():
    try:
        return OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=OPENAI_API_KEY
        )
    except Exception as e:
        st.error(f"Error creating embedding function: {str(e)}")
        return None

# Format documents into text
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Cache-safe vectorstore loader
@st.cache_resource
def load_cached_vectorstore(vectorstore_path, _embedding_function):
    return Chroma(persist_directory=vectorstore_path, embedding_function=_embedding_function)

# Vectorstore creation
def create_vectorstore(chunks, embedding_function, vectorstore_path):
    try:
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
        unique_ids = set()
        unique_chunks = []

        for chunk, id in zip(chunks, ids):
            if id not in unique_ids:
                unique_ids.add(id)
                unique_chunks.append(chunk)

        vectorstore = Chroma.from_documents(
            documents=unique_chunks,
            embedding=embedding_function,
            persist_directory=vectorstore_path
        )
        vectorstore.persist()
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None

# Document processing logic
def process_document(uploaded_file, chunk_size, chunk_overlap):
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()

        # Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " "]
        )
        chunks = text_splitter.split_documents(pages)

        embedding_function = get_embedding_function()
        if embedding_function is None:
            return None

        vectorstore_path = f"vectorstore_{uploaded_file.name.replace('.pdf', '')}"

        # Try to load existing vectorstore
        if os.path.exists(vectorstore_path):
            try:
                vectorstore = load_cached_vectorstore(vectorstore_path, embedding_function)
                return vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": num_chunks}
                )
            except Exception as e:
                st.warning(f"Could not load existing vectorstore. Rebuilding it. Reason: {e}")
                try:
                    shutil.rmtree(vectorstore_path)
                except Exception as remove_err:
                    st.error(f"Failed to delete vectorstore directory: {remove_err}")
                    return None

        os.makedirs(vectorstore_path, exist_ok=True)

        vectorstore = create_vectorstore(chunks, embedding_function, vectorstore_path)
        if vectorstore is None:
            return None

        os.unlink(tmp_file_path)

        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": num_chunks}
        )
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None

# RAG prompt template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following 
pieces of retrieved context to answer the question. If you don't know the 
answer, say that you don't know. Don't make up anything.

Context:
{context}

Question: {question}

Please provide a clear and concise answer based on the context above.
"""

# Create the RAG chain
def create_rag_chain(retriever):
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return (
        {
            "context": lambda x: format_docs(retriever.get_relevant_documents(x["question"])),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# Streamlit main interface
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing document..."):
        retriever = process_document(uploaded_file, chunk_size, chunk_overlap)

    if retriever is not None:
        rag_chain = create_rag_chain(retriever)

        question = st.text_input("Ask a question about the document:", key="question_input")

        if question:
            with st.spinner("Generating answer..."):
                try:
                    response = rag_chain.invoke({"question": question})
                    st.write("### Answer")
                    st.write(response)

                    with st.expander("Show Retrieved Context"):
                        chunks = retriever.get_relevant_documents(question)
                        for i, chunk in enumerate(chunks, 1):
                            st.write(f"#### Chunk {i}")
                            st.write(chunk.page_content)
                            st.write("---")
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
