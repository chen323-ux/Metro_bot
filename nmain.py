from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
import tempfile
import numpy as np

def main():



    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(page_title="Metro TransitTech ChatBot")
    st.header("TransitTech's ChatBot")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)

            # Save DataFrame to a temporary CSV file
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
                csv_path = tmp_file.name
                df.to_csv(csv_path, index=False)

            agent = create_csv_agent(
                OpenAI(temperature=0),
                csv_path,
                verbose=True
            )

            user_question = st.text_input("Ask a question about your CSV: ")

            if user_question is not None and user_question != "":
                with st.spinner(text="In progress..."):
                    st.write(agent.run(user_question))

        except Exception as e:
            st.error("API Key error.")
            st.error(str(e))


    df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

    st.map(df)





if __name__ == "__main__":
    main()


