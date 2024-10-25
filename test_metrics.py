import time
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


import os

# Load environment variables (like API keys)
load_dotenv()

# Ensure Google API key is set
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]




def test_query_latency(qa_chain, query):
    start_time = time.time()  # Start timing outside of the loop
    try:
        response = qa_chain.invoke(query)  # Invoke the query
        latency = time.time() - start_time  # Calculate latency
        return latency, response
    except Exception as e:  # Catch a general exception or a specific one if you know
        print(f"An error occurred: {e}")
        return None, None  # Return None for both latency and response in case of an error

# Throughput Test (run multiple queries and track how many per second)
def test_throughput(qa_chain, queries, duration=30):
    count = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        for query in queries:
            qa_chain.run(query)
            count += 1

    total_time = time.time() - start_time
    throughput = count / total_time
    print(f"Throughput: {throughput:.2f} queries per second over {duration} seconds.")
    return throughput

# Hallucination Rate Test (manually check if response is relevant)
def test_hallucination_rate(qa_chain, queries, relevant_responses):
    hallucination_count = 0
    total_queries = len(queries)

    for i, query in enumerate(queries):
        response = qa_chain.run(query)
        print(f"Query: {query}\nResponse: {response}")
        # Compare the response to a list of relevant, factual answers
        if response not in relevant_responses[i]:
            hallucination_count += 1

    hallucination_rate = (hallucination_count / total_queries) * 100
    print(f"Hallucination rate: {hallucination_rate:.2f}%")
    return hallucination_rate

# Set up the system (same as your main app)
def setup_system(pdf_path):
    # Read the PDF
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Splitting Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    # Embedding with Google Generative AI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")

    # Create vector store
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

    # Save the FAISS vector store (optional for repeated testing)
    faiss_index_path = "test_faiss_index"
    VectorStore.save_local(faiss_index_path)

    # Load FAISS vector store
    VectorStore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

    # Set up Gemini model for answering queries
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-002",
        temperature=0.7,
        max_output_tokens=512
    )

    # Set up retrieval-based question answering chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=VectorStore.as_retriever()
    )

    return qa_chain

if __name__ == "__main__":
    # Path to your PDF file
    pdf_path = "financial_document.pdf"  # Update with your file

    # Setup the QA system
    qa_chain = setup_system(pdf_path)

    # Define test queries and responses based on the financial document
    queries_and_responses = [
        {
            "query": "What is the name of the company in the financial report?",
            "response": "The name of the company is ABC Corp."
        },
        {
            "query": "What was the total revenue for ABC Corp. in fiscal year 2023?",
            "response": "The total revenue for ABC Corp. in fiscal year 2023 was $10,000,000."
        },
        {
            "query": "What is the net income reported for ABC Corp. for the fiscal year 2023?",
            "response": "The net income reported for ABC Corp. for the fiscal year 2023 was $1,500,000."
        },
        {
            "query": "What is the gross profit margin for ABC Corp.?",
            "response": "The gross profit margin for ABC Corp. is 40%."
        },
        {
            "query": "What is the EBITDA for ABC Corp.?",
            "response": "The EBITDA for ABC Corp. is $2,000,000."
        },
        {
            "query": "What is the earnings per share (EPS) for ABC Corp.?",
            "response": "The earnings per share (EPS) for ABC Corp. is $3.5."
        },
        {
            "query": "What is the dividend per share declared by ABC Corp.?",
            "response": "The dividend per share declared by ABC Corp. is $1.2."
        },
        {
            "query": "Who is the CEO of ABC Corp.?",
            "response": "The CEO of ABC Corp. is John Doe."
        },
        {
            "query": "What are the total assets of ABC Corp.?",
            "response": "The total assets of ABC Corp. are $10,200,000."
        },
        {
            "query": "What is the total liabilities amount for ABC Corp.?",
            "response": "The total liabilities amount for ABC Corp. is $5,500,000."
        },
        {
            "query": "What is the equity reported in the balance sheet?",
            "response": "The equity reported in the balance sheet for ABC Corp. is $4,700,000."
        },
        {
            "query": "What was the operating income for ABC Corp. in 2023?",
            "response": "The operating income for ABC Corp. in 2023 was $2,500,000."
        },
        {
            "query": "What is the cost of goods sold (COGS) for ABC Corp.?",
            "response": "The cost of goods sold (COGS) for ABC Corp. is $6,000,000."
        },
        {
            "query": "What is the cash flow from operations for ABC Corp.?",
            "response": "The cash flow from operations for ABC Corp. is $2,100,000."
        },
        {
            "query": "What is the net cash flow for ABC Corp.?",
            "response": "The net cash flow for ABC Corp. is $400,000."
        },
        {
            "query": "What is the current ratio for ABC Corp.?",
            "response": "The current ratio for ABC Corp. is 2.05."
        },
        {
            "query": "What is the quick ratio for ABC Corp.?",
            "response": "The quick ratio for ABC Corp. is 1.75."
        },
        {
            "query": "What is the debt-to-equity ratio for ABC Corp.?",
            "response": "The debt-to-equity ratio for ABC Corp. is 0.74."
        },
        {
            "query": "What is the return on assets (ROA) for ABC Corp.?",
            "response": "The return on assets (ROA) for ABC Corp. is 14.7%."
        },
        {
            "query": "What is the return on equity (ROE) for ABC Corp.?",
            "response": "The return on equity (ROE) for ABC Corp. is 31.9%."
        },
        {
            "query": "What is the price-to-earnings ratio for ABC Corp.?",
            "response": "The price-to-earnings ratio for ABC Corp. is 18."
        },
        {
            "query": "What is the market capitalization of ABC Corp.?",
            "response": "The market capitalization of ABC Corp. is $30,000,000."
        },
        {
            "query": "What is the cash flow from investing activities?",
            "response": "The cash flow from investing activities for ABC Corp. is -$1,200,000."
        },
        {
            "query": "What is the cash flow from financing activities?",
            "response": "The cash flow from financing activities for ABC Corp. is -$500,000."
        },
        {
            "query": "What is the stock performance year-to-date (YTD)?",
            "response": "The stock performance for ABC Corp. is +8% YTD."
        },
        {
            "query": "What is the sector performance year-to-date (YTD)?",
            "response": "The sector performance year-to-date (YTD) is +5%."
        },
        {
            "query": "What is ABC Corp.'s growth strategy?",
            "response": "ABC Corp.'s growth strategy involves expansion into new markets."
        },
        {
            "query": "What are some of the risk factors for ABC Corp.?",
            "response": "Some risk factors for ABC Corp. include economic downturn and supply chain disruptions."
        },
        {
            "query": "What is the outlook for ABC Corp. next fiscal year?",
            "response": "The outlook for ABC Corp. is positive, with a projected 10% revenue growth next fiscal year."
        },
        {
            "query": "What are the operating expenses for ABC Corp.?",
            "response": "The operating expenses for ABC Corp. are $1,500,000."
        },
        {
            "query": "What is the interest expense reported for ABC Corp.?",
            "response": "The interest expense reported for ABC Corp. is $300,000."
        },
        {
            "query": "What is the income before taxes for ABC Corp.?",
            "response": "The income before taxes for ABC Corp. is $2,200,000."
        },
        {
            "query": "What is the gross profit for ABC Corp.?",
            "response": "The gross profit for ABC Corp. is $4,000,000."
        },
    ]

    # Simulated query latency testing
    for query_response in queries_and_responses:
        query = query_response['query']
        expected_response = query_response['response']

        # Simulating query processing delay
        latency, response = test_query_latency(qa_chain, query)
        print(f"Expected: {expected_response}\nObtained: {response}\nLatency: {latency:.2f} seconds\n")
        time.sleep(0.5)  # 500ms delay
