import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers, HuggingFaceHub

from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.storage import InMemoryStore

from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS,Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader

import os
import tempfile

def custom_prompt():
    custom_prompt_template = """Use the following pieces of information to answer the user's question about TH Rosenheim University.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    Question: {question}
    
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                                input_variables=['context', 'question'])
    return prompt

def files_retriever(embeddings):
    DB_FAISS = 'FAISS_vectorstore_files'
    db = FAISS.load_local(DB_FAISS, embeddings)
    retriever_files = db.as_retriever(search_type = 'similarity', search_kwargs = {'k':2})
    return retriever_files

def QA_retriever(embeddings):
    DB_FAISS_QA = 'FAISS_vectorstore_QA'
    db_qa = FAISS.load_local(DB_FAISS_QA, embeddings)
    retriever_QA= db_qa.as_retriever(search_type = 'similarity', search_kwargs = {'k':3})
    return retriever_QA

def merge_retriever(retriever_files, retriever_QA):
    
    merge_retriever = MergerRetriever(retrievers = [retriever_files, retriever_QA])
    
    return merge_retriever

def llm_model():
    HF_TOKEN = "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"
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
    llm_model = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": 0.3,"max_new_tokens":512})
    return llm_model

#query = "tell me about application process at Rosenheim university of Applied sciences:"
#result = qa_chain({"query": query}) #({'query': query})

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about TH Rosenheim University🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def set_css():
    st.markdown("""
        <style>
        .stText { /* This class name is an example, you'll need to inspect your app's HTML to find the correct class for your message containers */
            width: 100% !important;
            white-space: normal !important;
        }
        </style>
        """, unsafe_allow_html=True)


def display_chat_history(chain):
    set_css()
    reply_container = st.container()    # Inserts an invisible container into your app that can be used to hold multiple elements
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

def main():
    # Initialize session state
    initialize_session_state()
    st.image("THRosenheim.png")
    st.title("Welcome to TH Rosenheim Chatbot")

    llm = llm_model()
    
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5",
                                       model_kwargs={'device': 'cpu'}, encode_kwargs = {'normalize_embeddings': True})

    retriever1 = files_retriever(embeddings)  
    retriever2 = QA_retriever(embeddings)
    retriever = merge_retriever(retriever1, retriever2) 
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = custom_prompt()
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=retriever,
                                                 memory=memory,
                                                    combine_docs_chain_kwargs={"prompt": prompt})     
        
    display_chat_history(qa_chain)

if __name__ == "__main__":
    main()

                


        

    








