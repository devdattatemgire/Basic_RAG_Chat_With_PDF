import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os

# Sidebar Content
with st.sidebar:
    st.title("PDF Chat App")
    add_vertical_space(5)
    st.write("Upload your PDF(s) and chat with them!")

def main():
    st.header("Chat with PDF")

    # Custom CSS for chat messages and input box
    st.markdown("""
        <style>
        .chat-container {
            display: flex;
            flex-direction: column-reverse;  /* New message at the bottom */
            max-height: 400px;
            overflow-y: auto;
        }
        .chat-message {
            background-color: #333333;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            color: white;
        }
        .response-message {
            background-color: #4F4F4F;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            color: white;
        }
        .input-box {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            padding: 10px;
            background-color: white;
            border-top: 1px solid #ccc;
        }
        </style>
    """, unsafe_allow_html=True)

    # Load environment variables
    load_dotenv()

    # Ensure Google API key is set
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    # Initialize chat history and embeddings cache in session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    if 'embeddings_cache' not in st.session_state:
        st.session_state['embeddings_cache'] = {}

    # PDF file uploader
    pdf_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

    if pdf_files:
        combined_text = ""
        for pdf in pdf_files:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                combined_text += page.extract_text() or ""

        # Splitting Text into Chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text=combined_text)

        # Use Google Generative AI Embeddings for embedding the document
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")

        # Create vector store
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Save the FAISS vector store to disk
        store_name = "combined_documents"
        faiss_index_path = f"{store_name}_faiss_index"
        VectorStore.save_local(faiss_index_path)

        # Store the embeddings in the cache
        st.session_state['embeddings_cache'][store_name] = embeddings

        # Load the FAISS vector store from disk with dangerous deserialization allowed
        VectorStore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

        # Display the chat container and messages, but only the new query-response pair will appear on each send
        st.write("<div class='chat-container'>", unsafe_allow_html=True)
        for message in st.session_state['chat_history']:  # Loop through all messages
            st.markdown(f"<div class='chat-message'>{message['query']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='response-message'>{message['response']}</div>", unsafe_allow_html=True)
        st.write("</div>", unsafe_allow_html=True)

        # Fixed input box at the bottom for user queries
        with st.form("chat_form", clear_on_submit=True):
            query = st.text_input("Ask something about your PDFs:", key="input_query")
            submitted = st.form_submit_button("Send")

            if submitted and query:
                # Initialize Google Gemini LLM for generation
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash-002",
                    temperature=0.7,
                    max_output_tokens=512
                )

                # Set up the retrieval-based question answering chain
                retriever = VectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever
                )

                # Generate response based on the query
                response = qa_chain.run(query)

                # Save the query and response to chat history
                st.session_state['chat_history'].append({"query": query, "response": response})

                # Display only the newly added query-response pair with custom styles
                st.write("<div class='chat-container'>", unsafe_allow_html=True)
                # Apply inline styles for the query container
                st.markdown(
                    f"<div style='background-color: #A8E6CF; border-radius: 10px; padding: 10px; margin-bottom: 10px; color: black;'>{query}</div>",
                    unsafe_allow_html=True
                )
                # Apply inline styles for the response container
                st.markdown(
                    f"<div style='background-color: #DCEDC1; border-radius: 10px; padding: 10px; margin-bottom: 10px; color: black;'>{response}</div>",
                    unsafe_allow_html=True
                )
                st.write("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
