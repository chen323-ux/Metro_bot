# python 3.8 (3.8.16) or it doesn't work
# pip install streamlit streamlit-chat langchain python-dotenv
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
import pandas as pd
import tempfile
import numpy as np

from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()

    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # setup streamlit page
    st.set_page_config(
        page_title="ChatBot Integration",
        page_icon="🤖"
    )


def main():
    init()

    # CSV ChatBot
    csv_chat_agent = None

    st.header("ChatBot Integration 🤖")

    st.subheader("CSV ChatBot")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)

            # Save DataFrame to a temporary CSV file
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
                csv_path = tmp_file.name
                df.to_csv(csv_path, index=False)

            csv_chat_agent = create_csv_agent(
                OpenAI(temperature=0),
                csv_path,
                verbose=True
            )

            user_question_csv = st.text_input("Ask a question about your CSV: ")

            if user_question_csv is not None and user_question_csv != "":
                with st.spinner("In progress..."):
                    st.write(csv_chat_agent.run(user_question_csv))

        except Exception as e:
            st.error("Error processing CSV file.")
            st.error(str(e))

    st.subheader("Conversational Memory ChatBot")

    chat = ChatOpenAI(temperature=0)

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    # sidebar with user input
    with st.sidebar:
        user_input = st.text_input("Your message: ", key="user_input")

        # handle user input
        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking..."):
                response = chat(st.session_state.messages)
            st.session_state.messages.append(
                AIMessage(content=response.content))

    # display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')


if __name__ == '__main__':
    main()
