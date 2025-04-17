import streamlit as st
from flask import Flask, request, jsonify
from qa_chain import qa_chain  # Import the qa_chain function

st.set_page_config(page_title="RAG Chat Assistant 🤖", page_icon="💬")

# App title
st.title("💡 Ask Your Assistant")
st.markdown("Talk to your AI assistant and get accurate, personalized answers! ✨")

# Sidebar info
with st.sidebar:
    st.markdown("### ⚙️ Assistant Settings")
    st.markdown("Powered by `LangChain`, `Ollama`, and `ChromaDB`. ⚡")

# Chat history in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input area
user_input = st.text_input("📝 Type your question here...")

if st.button("Send 🚀") and user_input.strip():
    # Add user message to history
    st.session_state.chat_history.append(("user", user_input))

    # Run the RAG pipeline
    with st.spinner("Thinking... 🤔"):
        try:
            result = qa_chain(user_input)
            response_text = result["result"]
            st.session_state.chat_history.append(("assistant", response_text))
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.session_state.chat_history.append(("assistant", error_msg))

# Display conversation history
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 Assistant:** {message}")


