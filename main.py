from conversation import Conversation, Agent, Message
import streamlit as st
import logging
logging.basicConfig(level=logging.INFO)

agent_choice = st.selectbox('Choose a model', Agent)

st.title("Welcome to generic Wikipedia chatbot!")
conversation = Conversation.start(agent=Agent(agent_choice))


def message_to_chat_message(message: Message):
    return {'role': message.sender.value, 'content': message.text}


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = list(
        map(
            message_to_chat_message,
            conversation.history
        )
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = conversation.send(prompt)
    with st.chat_message(response.sender.value):
        st.markdown(response.text)
    st.session_state.messages.append(message_to_chat_message(response))
