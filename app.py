
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv
import os
from datetime import datetime

# Load .env
load_dotenv()

# Initialize SerpAPI Search Wrapper
search_wrapper = SerpAPIWrapper()

# Streamlit UI
st.set_page_config(page_title="AI Search Engine 🔎", page_icon="⚡")
st.title("LangChain Chatbot with SerpAPI")
st.markdown("Get faster, more reliable answers with real-time Google Search via SerpAPI.")

# Sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
api_key = api_key or os.getenv("GROQ_API_KEY")
model_choice = st.sidebar.selectbox("Choose LLM model", ["gemma2-9b-it", "llama-3.1-8b-instant"])

# Initialize LLM
llm = ChatGroq(groq_api_key=api_key, model_name=model_choice, streaming=False)

# Prompt and Chain
prompt = PromptTemplate.from_template("Based on the following web search result, answer the user question in detail:\n\n{question}")
chain = LLMChain(llm=llm, prompt=prompt)

# Cached SerpAPI search
@st.cache_data(show_spinner=False)
def cached_search(query: str) -> str:
    return search_wrapper.run(query)

# Session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! Ask me anything, and I’ll search it fast with SerpAPI.", "timestamp": str(datetime.now())}
    ]

# Display previous messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        st.caption(f"_Timestamp: {msg['timestamp']}_")

# Chat input
if user_input := st.chat_input("Ask a question..."):
    st.session_state["messages"].append({"role": "user", "content": user_input, "timestamp": str(datetime.now())})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        start_time = datetime.now()
        with st.spinner("Searching Google..."):
            web_result = cached_search(user_input)
            response = chain.run(question=web_result)
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()

        st.write(response)
        st.caption(f"_Response time: {response_time:.2f} seconds_")

        st.session_state["messages"].append({
            "role": "assistant",
            "content": response,
            "timestamp": str(datetime.now()),
            "response_time": response_time
        })

        with st.expander("🔎 Web Search Result Used"):
            st.markdown(web_result)
