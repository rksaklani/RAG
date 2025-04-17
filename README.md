# RAG
Use the  rag  where  we use langchain , ollama ,streamlit and upload your pdf you can chat with your pdf



##Create venv
python -m venv venv source venv/bin/activate # or venv\Scripts\activate on Windows

##Install dependencies
pip install -r requirements.txt

##firstely add your data file in pdf format
name of the file should be --- mypdf.pdf

###Generate Vector DB (only once) it create the folder of the db_folder
python -c "from vectorstore import load_vectorstore; load_vectorstore()"

###Start the server
streamlit run app.py
