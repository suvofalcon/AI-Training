import os
from dotenv import load_dotenv
import openai
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

if __name__ == "__main__":
    print("Ingesting...")

    # load the text document to a langchain document
    loader = TextLoader("./mediumblog1.txt")
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents=document)
    print(f"Created {len(texts)} chunks")

    # Initialize the OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

    print("Passing to Vector Store...")
    PineconeVectorStore.from_documents(documents=texts, embedding=embeddings, index_name=os.environ["PINECONE_INDEX_NAME"])
    print("finish")
