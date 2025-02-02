import os
import openai
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub


load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

if __name__ == "__main__":

    # Get the path of the data
    dataset_path = os.path.join(os.environ['DATASETS_PATH'], 'ChainOfThoughtReasoning.pdf')
    loader = PyPDFLoader(file_path=dataset_path)
    # load the document
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)

    # We can persist the in memory vector store locally to our disk
    vectorstore.save_local("faiss_index_react")

    # we can load the vector store from our local and use that as the vector store
    new_vectorstore = vectorstore.load_local(folder_path="faiss_index_react", embeddings=embeddings,
                                             allow_dangerous_deserialization=True)

    '''We will now use Retrieval Chain'''

    retrieval_qa_chain_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm=ChatOpenAI(model='gpt-4o-mini'),
                                                      prompt=retrieval_qa_chain_prompt)

    retrieval_chain = create_retrieval_chain(retriever=new_vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)

    result = retrieval_chain.invoke({"input": "Give me the gist of ReAct in three sentences"})

    print(result['answer'])
