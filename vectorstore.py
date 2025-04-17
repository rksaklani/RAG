from langchain_community.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vectorstore(persist_directory="db_folder"):
    file_path = "mypdf.pdf"
    # loader = TextLoader(file_path)
    loader = PyPDFLoader(file_path)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = text_splitter.split_documents(document)

    embedding_obj = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_obj,
        persist_directory=persist_directory
    )

    vector_db.persist()
    return None  # To avoid keeping in memory

def load_retriever(persist_directory="db_folder"):
    embedding_obj = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_obj
    )
    return vector_db.as_retriever(search_kwargs={"k": 2})
