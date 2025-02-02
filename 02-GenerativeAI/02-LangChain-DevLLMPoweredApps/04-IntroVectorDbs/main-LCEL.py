import os
from dotenv import load_dotenv
import openai
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub

'''This source code demonstrates the retrieval using LCEL'''

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


if __name__ == "__main__":
    print("Retrieving..")

    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    llm_model = ChatOpenAI(model='gpt-4o-mini')

    # Lets try out a generic query
    query = "What is Pinecone in Machine Learning?"
    print(f"WHEN THE QUERY IS SENT TO OPEN AI DIRECTLY - {query}")
    chain = PromptTemplate.from_template(template=query) | llm_model
    result = chain.invoke(input={})
    print(result.content)

    vectorstore = PineconeVectorStore(index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings)
    # Standard Prompt downloaded from langchain hub - specifically written to reduce hallucinations
    retrieval_qa_chat_prompt = hub.pull('langchain-ai/retrieval-qa-chat')
    # Combine the documents chain
    combine_docs_chain = create_stuff_documents_chain(llm=llm_model, prompt=retrieval_qa_chat_prompt)
    # create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)

    # Now lets pass the same query
    print(f"WHEN THE SAME QUERY IS SENT VIA CHUNKS FROM VECTOR DATABASE - {query}")
    result = retrieval_chain.invoke(input={"input": query})
    print(result['answer'])
