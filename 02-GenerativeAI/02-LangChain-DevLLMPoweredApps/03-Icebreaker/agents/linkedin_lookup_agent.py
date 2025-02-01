import os
import openai
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from tools.tools import get_profile_url_tavily

openai.api_key = os.environ["OPENAI_API_KEY"]

def lookup(name: str) -> str:

    ''' Function which will take a name and get back the linkedin profile url of that name'''
    
    # Initialize the LLM Model
    llm_model = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
    
    # Initialize the template
    template = """Given the full name - {name_of_person}, please get me a link of their linkedin profile page.
                Your answer should contain only a URL"""

    prompt_template = PromptTemplate(input_variables=['name_of_person'], template=template)
   

    # All the tools that search agent is going to be using 
    tools_for_agent = [
        Tool(
        name="Crawl Google 4 linkedin profile page",
        func=get_profile_url_tavily,
        # this field is very important as lot of LLM reasoning happens from understanding this field
        description="useful when we need to get the LinkedIn Profile Page"
        )
    ]
    
    # define the react prompt - we will use a predefined prompt from the langchain hub
    react_prompt = hub.pull("hwchase17/react")

    # we create the react agent
    agent = create_react_agent(llm=llm_model, tools=tools_for_agent, prompt=react_prompt)
    # then we define the agent runtime, which is going to receive the agent, the list of tools and run it
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)
    
    # now we invoke the agent
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )
    
    linkedin_profile_url = result["output"]
    return linkedin_profile_url


if __name__ == "__main__":
    linkedin_url = lookup(name="Eden Marco Udemy")
    print(linkedin_url)
