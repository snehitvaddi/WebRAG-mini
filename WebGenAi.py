import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.title("Web GPT Mini: Get Answers from any website 📈")
st.sidebar.title("URLs")

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

llm = ChatOpenAI(model="gpt-4o", temperature=0.9, max_tokens=500)
embeddings = OpenAIEmbeddings()

# Initialize session state to store intermediate data
if 'docs' not in st.session_state:
    st.session_state.docs = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if process_url_clicked:
    urls = [i for i in urls if i]
    loader = UnstructuredURLLoader(urls=urls)

    main_placeholder.text("Opening URL...Started...✅✅✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    st.session_state.docs = text_splitter.split_documents(data)

    main_placeholder.text("Embedding...Started...✅✅✅")
    st.session_state.vectorstore = FAISS.from_documents(st.session_state.docs, embeddings)

    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    st.session_state.vectorstore.save_local("vectorstore")

# Only show the query input if URLs have been processed
if st.session_state.vectorstore is not None:
    query = st.text_input("Enter question")

    if query:
        vecstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vecstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        
        st.header("Answer")
        st.write(result["answer"])
        
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
