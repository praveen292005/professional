import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set your API key in the .env file.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


def create_vector_store(file_path: str) -> FAISS:
    """Ingest a text document and create a FAISS vector store."""
    try:
        loader = TextLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store
    except FileNotFoundError:
        raise FileNotFoundError(f"The book file '{file_path}' was not found.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while creating the vector store: {e}")


def create_rag_chain(vector_store: FAISS):
    """Create a RAG chain to handle queries."""
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

    # Create the prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context. If the answer
        is not in the context, say that you don't have enough information from the book.

        Context: {context}
        Question: {input}
        """
    )

    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)

    # Create the retriever and the retrieval chain
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


def user_input(user_question: str, rag_chain):
    """Process user input and return a response."""
    try:
        response = rag_chain.invoke({"input": user_question})
        return response["answer"]
    except Exception as e:
        return f"An error occurred while generating the response: {e}"