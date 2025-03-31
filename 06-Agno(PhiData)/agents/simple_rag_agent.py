from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType

def create_knowledge_base():

    knowledge_base = PDFUrlKnowledgeBase(
        urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        # Use LanceDB as the vector database
        vector_db=LanceDb(
            table="recipes",
            uri="resources/lancedb",
            search_type=SearchType.vector,
            embedder=OpenAIEmbedder(model="text-embedding-3-small")
        )
    )
    return knowledge_base

def create_rag_agent():

    knowledge = create_knowledge_base()
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        # Add the knowledgebase to the agent
        knowledge_base=knowledge,
        show_tool_calls=True,
        markdown=True
    )
    return agent

if __name__ == "__main__":
    rag = create_rag_agent()
    rag.print_response("How do I make chicken and galangal in coconut milk soup", stream=True)
