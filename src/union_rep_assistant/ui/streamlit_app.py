# TODO: https://docs.streamlit.io/develop/concepts/connections/secrets-management
# TODO: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
#  add secret management.  The point is to send a link and the person can use it.

import os

import streamlit as st

from union_rep_assistant.brain import UnionRep
from union_rep_assistant.ui.validation import is_valid_api_key
from union_rep_assistant.constants import SECURITY_CONTRACT_PATH

openai_api_key = os.getenv("OPENAI_API_KEY")
# Streamlit Sidebar for API Key
if not openai_api_key:
    with st.sidebar:
        openai_api_key = st.text_input(
            "OpenAI API Key",
            key="chatbot_api_key",
            type="password",
        )
        st.markdown(
            "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        )
    # TODO: need to figure out how to add secrets because I want to just pass this chat to charles.

# Check if API Key is Entered
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

if not is_valid_api_key(openai_api_key):
    st.stop()


union_rep = UnionRep(
    model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key, top_k=3
)

# Streamlit UI
st.title("💬 Union Rep Assistant")
st.caption(f"using data source:  {str(SECURITY_CONTRACT_PATH)}")

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

# TODO: error handling for when api key has run out of money?
