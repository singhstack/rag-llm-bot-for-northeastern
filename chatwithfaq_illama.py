import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import os
import time
import json
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.document_loaders import JSONLoader

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="FAQ Chat")

#!echo -e 'GOOGLE_API_KEY=AIzaSyBdlnKJKDXsxqTPJ8IZh4nmg_95CLRxy5Q' > .env


def get_texts():
    file_path = '/Users/smoothoperator/Documents/GitHub/rag-llm-bot-for-northeastern/dataset/faqs.json'

    data = json.loads(Path(file_path).read_text())

    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.',
        text_content=False)

    # load documents
    data = loader.load()
    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    texts = text_splitter.split_documents(data)
    return texts

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    texts_str = []
    for text in text_chunks:
        texts_str.append(str(text))
    
    vector_store = FAISS.from_texts(texts_str, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    #FAISS.allow_dangerous_deserialization = True
    new_db = FAISS.load_local("faiss_index", embeddings,  allow_dangerous_deserialization=True)

    #st.success("Data loaded successfully!")
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    #print(response)
    #st.write("Reply: ", response["output_text"])
    for word in response['output_text'].split():
        yield word + " "
        time.sleep(0.05)

#fix the flow
#clear the input dialogie post "submission"


def main():

    st.title("Echo Bot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.text_input("Ask a Question.")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})
    
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(user_input(user_question))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def main_old():
    st.set_page_config("FAQ Chat")
    st.header("Chat with student employment FAQs using Gemini💁")


    user_question = st.text_input("Ask a Question.")

    if user_question:
        user_input(user_question)


    with st.sidebar:
        st.title("Menu:")
        #pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                #raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_texts()
                get_vector_store(text_chunks)
                st.write("Data loaded successfully!")
                st.success("Done")

if __name__ == "__main__":
    main()