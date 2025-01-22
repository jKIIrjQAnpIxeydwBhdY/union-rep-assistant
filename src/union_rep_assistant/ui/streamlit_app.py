from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from union_rep_assistant.brain import UnionRep
from union_rep_assistant.ui.validation import is_valid_api_key


def initialize_union_rep(openai_api_key):
    LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
    EMBEDDINGS = OpenAIEmbeddings(openai_api_key=openai_api_key)

    base_path = Path(__file__).parent  # assumes faiss will always be here
    cached_vector_store = FAISS.load_local(
        base_path / "faiss_index", EMBEDDINGS, allow_dangerous_deserialization=True
    )

    union_rep = UnionRep(cached_vector_store, LLM)
    return union_rep


# Streamlit Sidebar for API Key
with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key",
        key="chatbot_api_key",
        type="password",
    )
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")


# Check if API Key is Entered
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

if not is_valid_api_key(openai_api_key):
    st.stop()


union_rep = initialize_union_rep(openai_api_key)

# Streamlit UI
st.title("💬 Union Rep Helper")

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Display Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle User Input
if user_input := st.chat_input():
    # Add User Message to Chat History
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    try:
        # Generate Response Using union_rep.ask
        response = union_rep.ask(user_input)  # Fetch the response from the LLM
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
    except Exception as e:
        st.error(f"Error: {str(e)}")
