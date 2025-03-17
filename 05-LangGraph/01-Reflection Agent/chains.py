from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Build the reflection Prompt and generation prompt

# Agent will introspect and reflect using this prompt again and again
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user."
            "Always provide detailed recommendations, including requests for length, virality, style etc."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
# Agent will generate/revise response after getting the reflection prompt
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            "Generate the best twitter post possible for the user's request."
            "If the user provides critique, respond with a revised version of your previous attempts."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define the Chains

llm = ChatOpenAI()
generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
