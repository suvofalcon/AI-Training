import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Initialize the embeddings
load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

'''
All ingestion of documents will happen inside this function
'''
def ingest_docs():
    data_path = os.path.join((os.environ['DATASET_PATH']), 'api.python.langchain.com/en/latest')
    loader = ReadTheDocsLoader(data_path)
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs","https:/")
        doc.metadata.update(({"source": new_url}))

    # add these to the Vector Store
    print(f"Going to add {len(documents)} documents to Pinecone vector store")
    PineconeVectorStore.from_documents(documents=documents,
                                       embedding=embeddings,
                                       index_name="langchain-doc-index")
    print("**** Loading to Vector Store complete...****")


if __name__ == '__main__':
    ingest_docs()





