import streamlit as st
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

def load_document(file):
    name, extension = os.path.splitext(file)
    if extension == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == ".txt":
        from langchain_community.document_loaders import TextLoader
        print(f'Loading {file}')
        loader = TextLoader(file)
    else:
        print(f'File extension {extension} not supported.')
        return None
    data = loader.load()
    return data

def chunk_data(data, chunk_size = 256, chunk_overlap = 20):
    from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if isinstance(data, list):
        chunks = text_splitter.split_documents(data)
    else:
        chunks = text_splitter.split_text(data)
    return chunks

def create_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
        )
    return vector_store

def ask_and_get_answer(vector_store, q, k=3):
    from langchain_groq import ChatGroq
    from langchain_classic.chains import ConversationalRetrievalChain

    llm = ChatGroq(
        model='llama-3.3-70b-versatile',
        temperature=1
    )

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    response = chain.invoke({"question": q, "chat_history": []})

    return response

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.header("LLM Q & A Application")

    with st.sidebar:
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
        chunk_size = st.number_input('Chunk Size', min_value = 100, max_value = 2048, value = 256, on_change=clear_history)
        k = st.number_input('K', min_value = 1, max_value = 20, value = 3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, "wb") as f:
                    f.write(bytes_data)
                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')
                vector_st = create_embeddings(chunks)

                st.session_state.vs = vector_st
                st.success('File uploaded, chunked and embedded successfully.')

    q = st.text_input('Ask a question about the content of your file: ')
    if 'history' not in st.session_state:
        st.session_state.history = ''
    if q:
        if 'vs' in st.session_state:
            vector_st = st.session_state.vs

            response = ask_and_get_answer(vector_st, q, k)
            answer_text = response['answer']

            st.text_area("LLM Answer: ", value=answer_text)

            new_entry = f"Q: {q}\nA: {answer_text}"
            st.session_state.history = f"{new_entry}\n{'-' * 100}\n{st.session_state.history}"

    st.divider()
    st.text_area(label='Chat History', value=st.session_state.history, height=400)







# use 'streamlit run chat_with_documents.py' to run

