import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents import create_csv_agent
import os
import datetime
from streamlit_mic_recorder import mic_recorder, speech_to_text

import constants 

# Format the timestamp
now = datetime.datetime.now()
formatted_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
formatted_timestamp_ymd = now.strftime("%Y-%m-%d")

os.environ["OPENAI_API_BASE"] = constants.BASE_URL # Replace with your URL  - Please check to see if we need this ?
os.environ["OPENAI_API_KEY"] = constants.APIKEY
os.environ["OPENAI_API_VERSION"] = "2023-03-15"
GPT_ENGINE = "gpt-35-turbo"

# Initialize session state  
if 'text_received' not in st.session_state:
    st.session_state['text_received'] = []

st.set_page_config(layout="wide")
st.title("Technology Service (TS) Copilot MVP")
st.subheader("GenAI Data Analytics and Insight - " + formatted_timestamp)
pic = "https://gena.com/static/media/genAiLogo.c1404741ad8a325238f390013e2cef30.svg"  
st.image(pic, width=100)

# Define tabs
main_csv_agent_tab, text_qna_tab = st.tabs(["CSV Agent", "Text QNA"])

def run_agent(fp, text):
    agent_csv = create_csv_agent(
        AzureChatOpenAI(deployment_name=GPT_ENGINE, temperature=0, max_tokens=1000),
        handle_parsing_errors=True,
        path=fp,
        max_iterations=20,
        verbose=True
    )
    response = agent_csv.run(text)
    return response

# CSV Agent tab section
with main_csv_agent_tab:
    # Upload CSV file
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

    # Create a dropdown selection in the sidebar
    selected_option = st.sidebar.selectbox("Select prompt:",
        ["Show csv columns.", "What is the row count?"]
    )

    # Get user input
    user_input = st.sidebar.text_area("Enter a prompt:", selected_option)

    st.sidebar.write("[MLFlow Link](http://127.0.0.1:5000)")

    # Label response
    st.subheader("Response:")

    # Run agent and display response
    if uploaded_file and user_input:
        response1 = run_agent(user_input,uploaded_file)
        st.write(response1)
    else:
        st.write("Please upload a CSV file and enter a prompt.")

# Text QNA tab section
user_input1 = ""
response1 = ""
with text_qna_tab:
    chat = AzureChatOpenAI(
        azure_endpoint=os.environ["OPENAI_API_BASE"],  # Use azure_endpoint here
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_version="2023-05-15",
        deployment_name="gpt-35-turbo",
        model_version="0613",
        azure_deployment="gpt-35-turbo",
    )


    sys_persona = "A highly experienced and knowledgeable Enterprise Architect with an extensive background..."
    # Truncated for brevity
    sys_prompt = st.text_area(label="Enter system persona prompt", value=sys_persona)
    sys_prompt = sys_prompt + " Respond I DON'T KNOW if answer is not available."

    if st.session_state['text_received']:
        for text in st.session_state['text_received']:
            st.text_input(label="Question-microphone:", value=text)
            user_input1 = text

    c1, c2 = st.columns(2)
    with c1:
        user_input1 = st.text_input("Question-keyboard:", user_input1)
    with c2:
        text = speech_to_text(start_prompt="Convert speech to text:", language='en', use_container_width=False, just_once=True, key='STT')
        if text:
            st.session_state['text_received'].append(text)
        for text in st.session_state['text_received']:
            st.text_input(label="Question-microphone:", value=text)
            user_input1 = text
        st.write("Record your voice, and play the recorded audio:")
        audio = mic_recorder(start_prompt="⏺️", stop_prompt="⏹️", key='recorder')
        if audio:
            st.audio(audio['bytes'])

    if user_input1:
        if user_input1.lower() == "exit":
            st.stop()

        user_input1 = sys_prompt + user_input1
        results = chat.invoke(user_input1)
        # Label response
        st.subheader("Response:")
        st.markdown(results.content)
        response1 = results.content

# Define the run_agent function

# Start an MLflow run
import mlflow
with mlflow.start_run():
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment("TS Copilot Test Run: " + formatted_timestamp_ymd)

    developer = 'RJohnson'
    mlflow.log_param("coder", developer)
    mlflow.log_param("model", GPT_ENGINE)
    mlflow.log_param("input user", user_input1)
    mlflow.log_param("response", response1[:500])  # MLFlow restricts parameter to 500 characters

    uip_tmp = "User Input: " + user_input1 + "\nResponse: " + response1
    mlflow.log_text(uip_tmp, artifact_file="user_input_response.txt")

    mlflow.end_run()
