import streamlit as st
import requests

# Define FastAPI server URL
SERVER_URL = "http://localhost:8000"

# Streamlit app
def main():
    st.title("Document Chat App")

    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "csv", "txt", "xls", "xlsx", "docx"])
    if uploaded_file is not None:
        st.write("File uploaded successfully!")

        # Upload file to FastAPI server
        upload_response = requests.post(f"{SERVER_URL}/uploadfile/", files={"file": uploaded_file})
        if upload_response.status_code == 200:
            st.success(upload_response.json()["message"])
        else:
            st.error(f"Error uploading file: {upload_response.text}")

    if st.button("Load Documents"):
        # Load documents from FastAPI server
        load_response = requests.get(f"{SERVER_URL}/load_docs/")
        if load_response.status_code == 200:
            st.success("Documents loaded successfully!")
        else:
            st.error(f"Error loading documents: {load_response.text}")

    st.header("Ask a Question")
    question = st.text_input("Enter your question")
    if st.button("Get Answer"):
        if question:
            # Get answer from FastAPI server
            answer_response = requests.get(f"{SERVER_URL}/answer/", params={"question": question})
            if answer_response.status_code == 200:
                answer = answer_response.json()["answer"]
                st.success(f"Answer: {answer}")
            else:
                st.error(f"Error getting answer: {answer_response.text}")
        else:
            st.warning("Please enter a question")


if __name__ == "__main__":
    main()
