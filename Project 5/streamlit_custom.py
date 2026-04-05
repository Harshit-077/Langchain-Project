import streamlit as st
from streamlit_chat import message
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langchain_classic.schema import SystemMessage, HumanMessage, AIMessage
load_dotenv(find_dotenv(), override=True)

st.set_page_config(
    page_title="Your Custom Assistant",
    page_icon='👾'
)

st.subheader('Your Custom ChatGPT 👾')
chat = ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0.5
)

if 'messages' not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    system_message = st.text_input(label='System Role')
    user_prompt = st.text_input(label='Send a message')
    if system_message:
        if not any(isinstance(x, SystemMessage) for x in st.session_state.messages):
            st.session_state.messages.append(
                SystemMessage(
                    content = system_message
                )
            )

    if user_prompt:
        st.session_state.messages.append(
            HumanMessage(
                content = user_prompt
            )
        )
        with st.spinner('Working on your request ...'):
            response = chat.invoke(st.session_state.messages)

        st.session_state.messages.append(AIMessage(content = response.content))

# st.session_state.messages
# message('This is ChatGPT', is_user=False)
# message('This is the User', is_user=True)

if len(st.session_state.messages) >= 1:
    if not isinstance(st.session_state.messages[0], SystemMessage):
        st.session_state.messages.insert(0, SystemMessage(content="You are a helpful assistant."))


for i,msg in enumerate(st.session_state.messages[1:]):
    if i%2==0:
        message(msg.content, is_user=True, key=f'{i} + 😎')
    else:
        message(msg.content, is_user=False, key=f'{i} + 🤖')