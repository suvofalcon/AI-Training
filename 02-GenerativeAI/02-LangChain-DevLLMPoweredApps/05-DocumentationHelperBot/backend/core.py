import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

'''
This function will perform the Vector Database Search
'''
def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings)
    chat = ChatOpenAI(verbose=True, api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0, model="gpt-4o-mini")

    # Get the predefined retrieval qa chat prompt
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Now create the stuff documents chain - take the chain, augment with query and send that to llm
    stuff_documents_chain = create_stuff_documents_chain(llm=chat, prompt=retrieval_qa_chat_prompt)

    # Upon receiving the user's question and its history, we want to rephrase to a new standalone question
    # That question will hold all the information - for this we need a new prompt
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(llm=chat, retriever=docsearch.as_retriever(),
                                                             prompt=rephrase_prompt)

    qa = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain)

    # Now invoke the chain
    result = qa.invoke(input={"input": query, "chat_history": chat_history})

    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }
    return  new_result


if __name__ == '__main__':
    response = run_llm(query="What is a Langchain Chain?")
    print(response["result"])
