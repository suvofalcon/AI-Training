# Library imports
import os
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from third_parties.linkedin import scrape_linked_profile

openai.api_key = os.environ["OPENAI_API_KEY"]

summary_template = """
Given the LinkedIn information {information} about a person, please create:
1. A Short Summary
2. Two interesting facts about them
"""

summary_prompt_template = PromptTemplate(
    input_variables=["information"], template=summary_template
)

model = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

chain = summary_prompt_template | model

linkedin_data = scrape_linked_profile(linkedin_profile_url="https://www.linkedin.com/in/subhankarbhattacharya/")
response = chain.invoke(input={"information": linkedin_data})
print(response.content)

