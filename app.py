import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI  # Import Google Generative AI
from langchain.chains import RetrievalQA
import os

# Sidebar Content
with st.sidebar:
    st.title("Tamil PDF Chat App")
    add_vertical_space(5)
    st.write("")

def main():
    st.header("Chat with PDF")

    # Load environment variables
    load_dotenv()

    # Ensure Google API key is set
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]  # Ensure this is set in Streamlit secrets

    # Initialize a dictionary to cache embeddings
    embeddings_cache = {}

    # Upload PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        # Save the uploaded PDF name (optional)
        st.write(pdf.name)

        # Check if embeddings are already cached
        pdf_key = pdf.name  # Use the PDF name as a unique key

        if pdf_key in embeddings_cache:
            # Use cached embeddings
            embeddings = embeddings_cache[pdf_key]
            st.write("Using cached embeddings.")
        else:
            # Read the PDF
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Splitting Text into Chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # Use Google Generative AI Embeddings for embedding the document
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")

            # Create vector store
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

            # Save the FAISS vector store to disk
            store_name = pdf.name[:-4]
            faiss_index_path = f"{store_name}_faiss_index"
            VectorStore.save_local(faiss_index_path)  # Save the FAISS index to a file

            # Store the embeddings in the cache
            embeddings_cache[pdf_key] = embeddings

            st.write("Generated embeddings and saved vector store.")

        # Load the FAISS vector store from disk with dangerous deserialization allowed
        VectorStore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

        # Query input from user
        query = st.text_input("Ask something about your PDF:")

        if query:
            # Initialize Google Gemini LLM for generation (with model specified)
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-002",  # Ensure the correct model is used
                temperature=0.7,
                max_output_tokens=512
            )

            # Set up the retrieval-based question answering chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=VectorStore.as_retriever()
            )

            # Generate response based on the query
            response = qa_chain.run(query)
            st.write(response)

if __name__ == "__main__":
    main()
