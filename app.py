#use the flow from the video
#https://www.youtube.com/watch?v=o4ZhXSVuPyc&ab_channel=KrishNaik

# incorporate RAG components from the google colab
#https://colab.research.google.com/drive/1232WmmcCd152ffSUKty5Q9aMCnuhun8H?authuser=2#scrollTo=LuNyag9IXvPJ

from dotenv import load_dotenv
load_dotenv() ## loading all the environment variables

import streamlit as st
import os
import google.generativeai as genai

#RAG libs
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain.vectorstores import Chroma
import json
from pathlib import Path
from pprint import pprint


file_path = '/content/drive/My Drive/faqs.json'

data = json.loads(Path(file_path).read_text())

loader = JSONLoader(
    file_path=file_path,
    jq_schema='.',
    text_content=False)

# load documents
data = loader.load()
# split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


texts_str = []
for text in texts:
  texts_str.append(str(text))

vector_index = Chroma.from_texts(texts_str, embeddings).as_retriever()

prompt_template = """Answer the question as precisely as possible using the provided context. If the answer is
                    not contained in the context, say "answer not available in context" \n\n
                    Context: \n {context}?\n
                    Question: \n {question} \n
                    Answer:
                  """
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)




initial_context = #defined in initial_context.txt
chat = model.start_chat(history=[])


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

from langchain_google_genai import ChatGoogleGenerativeAI
model