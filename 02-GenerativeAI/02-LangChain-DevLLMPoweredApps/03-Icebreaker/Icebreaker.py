# Library imports
import os
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedin import scrape_linked_profile

openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize the llm model
model = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

def ice_break_with(name: str):

    # Get the linkedin username profile from the search string 
    linkedin_username_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linked_profile(linkedin_profile_url=linkedin_username_url)

    # Define the prompt template
    summary_template = """
    Given the LinkedIn information {information} about a person, please create:
    1. A Short Summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )   

    # Define the chain of execution and invoke the chain
    chain = summary_prompt_template | model

    response = chain.invoke(input={"information": linkedin_data})
    return response


if __name__ == "__main__":
    response = ice_break_with(name="Eden Marco Udemy")
    print(response.content)
