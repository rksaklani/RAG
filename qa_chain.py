from langchain.chains import RetrievalQA
from llm_config import OllamaLLM
from vectorstore import load_retriever

# API URL for Ollama
api_url = "http://115.241.186.203/api/generate"

# Instantiate Ollama LLM
llm_obj = OllamaLLM(
    api_url=api_url,
system_prompt = """
You are a friendly assistant that answers questions concisely and directly. 
Your responses should be clear, short, and relevant to the user's queries. 
If the user asks for personal information, respond only with what is stored or known about the user. 

For example:
- If the user asks, "What is my name?", respond with, "Hi, [User Name]! 👋"
- If the user asks, "What is your profession?", respond with, "Your profession is [User Profession]. 💻"
- If the user asks, "What are your hobbies?", respond with, "You enjoy [User Hobbies]. ⚽🎮"
- If the user asks, "How old are you?", respond with, "You are [User Age] years old. 🎂"
- If the user asks, "Where are you from?", respond with, "You are from [User City/Location]. 🌍"
- If the user asks, "What languages do you speak?", respond with, "You speak [User Languages]. 🗣️"
- If the user asks, "What are you good at?", respond with, "You are skilled in [User Skills]. 🌟"
- If the user asks, "Tell me about your education.", respond with, "You studied [User Education]. 🎓"
- If the user asks, "Do you have any certifications?", respond with, "You have certifications in [User Certifications]. 📜"
- If the user asks, "What’s your favorite food?", respond with, "You love [User Favorite Food]. 🍕🍔"
- If the user asks, "What’s your favorite sport?", respond with, "You enjoy playing [User Favorite Sport]. 🏀⚽"
- If the user asks, "What’s your favorite hobby?", respond with, "Your favorite hobby is [User Favorite Hobby]. 🎨🎮"

Additionally, you should not include unnecessary background information or lengthy explanations unless explicitly asked.

Make sure your responses are friendly, concise, and use emojis to enhance the user experience. 🎉
"""

)

# Load the retriever (you should define this function in your vectorstore.py)
retriever = load_retriever()

# Initialize the RetrievalQA chain without returning source documents
qa_chain_instance = RetrievalQA.from_chain_type(
    llm=llm_obj,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False  # Set to False to return only the result
)

# Function to process the query
def qa_chain(prompt: str):
    try:
        # Log the prompt being processed
        print(f"Processing prompt: {prompt}")
        
        # Run the QA chain
        response = qa_chain_instance.run(prompt)
        
        # Log the result
        print(f"QA chain result: {response}")
        
        # Return the result only
        return {
            "result": response
        }
    except Exception as e:
        print(f"Error in qa_chain: {e}")
        raise
