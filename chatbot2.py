import streamlit as st
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from cachetools import cached, TTLCache
import os
from langdetect import detect

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-proj-jruZRi8TJjmA8BlyIqhKT3BlbkFJSAm4cRePs8ZC70Sh2wif"

# Custom CSS to change header color
custom_css = """
<h style="font-size:44px; color: #50BFE6">Extract Medical related info through MediBot</h>
"""
st.markdown(custom_css, unsafe_allow_html=True)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Let me know your query"}
    ]

# Create a cache with a Time-To-Live (TTL) of 1 hour
cache = TTLCache(maxsize=100, ttl=3600)

@st.cache_resource(show_spinner=False)
@cached(cache)
def load_data():
    try:
        with st.spinner(text="Loading and indexing the documents. This should take 1-2 minutes."):
            # loader = PyPDFDirectoryLoader("data")
            loader = Docx2txtLoader("data/Heart_Problem.docx")
            data = loader.load()
            print(data)

            chunk_size = 400
            chunk_overlap = 100
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = splitter.split_documents(data)

            embeddings = OpenAIEmbeddings()
            index = Chroma.from_documents(docs, embeddings)
            return index
        
    except Exception as e:
        st.error(f"An error occurred while loading and indexing the documents: {e}")
        return None

index = load_data()

if index:
    retriever = index.as_retriever()
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

    template = """You are an intelligent assistant trained to provide information and answer questions based on specific documents provided. Your responses should be informative, accurate, and relevant to the content of these documents. You should maintain a professional tone at all times. If a question is unrelated to the data provided, politely suggest that the information might not be available in the current data set. Now answer the question based on the context: {context}. Question: {question}"""

    prompt = ChatPromptTemplate.from_template(template)

    def get_chain():
        return (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

    chain = get_chain()

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve context based on the latest user question
                    user_question = st.session_state.messages[-1]["content"]
                    context_docs = retriever.get_relevant_documents(user_question)
                    context = "\n\n".join([doc.page_content for doc in context_docs])

                    # Detect language
                    language = detect(user_question)
                    if language == "bn":
                        # Update template for Bangla response
                        template_bn = """You are an intelligent assistant trained to provide information and answer questions based on specific documents provided. Your responses should be informative, accurate, and relevant to the content of these documents. Please respond in Bangla. You should maintain a professional tone at all times. If a question is unrelated to the data provided, politely suggest that the information might not be available in the current data set. Now answer the question based on the context: {context}. Question: {question}"""
                        prompt_bn = ChatPromptTemplate.from_template(template_bn)
                        chain_bn = (
                            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                            | prompt_bn
                            | model
                            | StrOutputParser()
                        )
                        chain_input = {"question": user_question, "context": context}
                        response = chain_bn.invoke(chain_input)
                    else:
                        # Prepare input for the chain with context in English
                        chain_input = {"question": user_question, "context": context}
                        response = chain.invoke(chain_input)

                    st.write(response)
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
