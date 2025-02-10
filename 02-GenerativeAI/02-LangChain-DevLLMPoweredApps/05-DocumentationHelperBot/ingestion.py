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

'''
This function will ingest documents using firecrawl loader
'''
def ingest_docs2():
    from langchain_community.document_loaders import FireCrawlLoader

    langchain_documents_base_urls = [
        "https://python.langchain.com/v0.2/docs/integrations/chat/",
        "https://python.langchain.com/v0.2/docs/integrations/llms/",
        "https://python.langchain.com/v0.2/docs/integrations/text-embedding/",
        "https://python.langchain.com/v0.2/docs/integrations/document_loaders/",
        "https://python.langchain.com/v0.2/docs/integrations/document_transformers/",
        "https://python.langchain.com/v0.2/docs/integrations/vectorstores/",
        "https://python.langchain.com/v0.2/docs/integrations/retrievers/",
        "https://python.langchain.com/v0.2/docs/integrations/tools/",
        "https://python.langchain.com/v0.2/docs/integrations/stores/",
        "https://python.langchain.com/v0.2/docs/integrations/llm_caching/",
        "https://python.langchain.com/v0.2/docs/integrations/graphs/",
        "https://python.langchain.com/v0.2/docs/integrations/memory/,"
        "https://python.langchain.com/v0.2/docs/integrations/callbacks/",
        "https://python.langchain.com/v0.2/docs/integrations/chat_loaders/",
        "https://python.langchain.com/v0.2/docs/integrations/concepts/"
    ]

    # Since this is lot of information, for demo purposes only taking the first url from the list and showing
    #langchain_documents_base_urls_mini = [langchain_documents_base_urls[0]]

    for url in langchain_documents_base_urls:
        print(f"FireCrawling {url=}")
        loader = FireCrawlLoader(
            url=url,
            mode="scrape",
        )
        docs = loader.load()
        print(f"Going to add {len(docs)} documents to Pinecone vector store")
        PineconeVectorStore.from_documents(embedding=embeddings, index_name="firecrawl-index", documents=docs)
        print("**** Loading {url) to Vector Store complete...****")

if __name__ == '__main__':
    ingest_docs2()





