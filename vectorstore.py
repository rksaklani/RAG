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



# import os
# from langchain_community.vectorstores import Chroma
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings



# def load_vectorstore(folder_path="data", persist_directory="db_folder"):
#     documents = []
#     for file_name in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, file_name)

#         if file_name.endswith(".pdf"):
#             loader = PyPDFLoader(file_path)
#         elif file_name.endswith(".txt"):
#             loader = TextLoader(file_path)
#         else:
#             continue

#         try:
#             docs = loader.load()
#             documents.extend(docs)
#         except Exception as e:
#             print(f"❌ Error loading {file_name}: {e}")

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
#     chunks = text_splitter.split_documents(documents)

#     embedding_obj = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     vector_db = Chroma.from_documents(
#         documents=chunks,
#         embedding=embedding_obj,
#         persist_directory=persist_directory
#     )

#     vector_db.persist()
#     print(f"✅ Vector store created with {len(chunks)} chunks.")
#     return None

# # ✅ Add this part
# def load_retriever(persist_directory="db_folder"):
#     embedding_obj = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_db = Chroma(
#         persist_directory=persist_directory,
#         embedding_function=embedding_obj
#     )
#     return vector_db.as_retriever(search_kwargs={"k": 2})
