import streamlit as st
from flask import Flask, request, jsonify
from qa_chain import qa_chain  # Import the qa_chain function

# app = Flask(__name__)

# @app.route("/v1/chat", methods=["POST"])
# def chat():
#     try:
#         data = request.json
#         user_prompt = data.get("instructions", "").strip()

#         if not user_prompt:
#             return jsonify({"error": "No instructions provided"}), 400

#         # Call the QA chain with the user prompt
#         response = qa_chain(user_prompt)

#         # Check if source documents are included or not
#         if "source_documents" in response:
#             return jsonify({
#                 "response": {
#                     "name": "Assistant",
#                     "content": response["result"],
#                     "sources": response["source_documents"]
#                 }
#             })
#         else:
#             return jsonify({
#                 "response": {
#                     "name": "Assistant",
#                     "content": response["result"]
#                 }
#             })

#     except Exception as e:
#         # Log the exception details to help debug
#         app.logger.error(f"Error occurred: {e}")
#         return jsonify({"error": str(e)}), 500




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





# if __name__ == "__main__":
#     app.run(debug=True, port=5000)

