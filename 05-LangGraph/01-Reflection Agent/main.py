from typing import List, Sequence
from dotenv import load_dotenv
#from langchain.chains.question_answering.map_reduce_prompt import messages

load_dotenv()

from constants import *
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from chains import generate_chain, reflection_chain

'''
Build the Graph
- define nodes
- define edges
- define conditional edge
- Compile Graph
- Visualize Graph
'''
# The generation node is going to receive the STATE as an input where here the STATE is just sequence of messages
def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})

# The refection node is also going to receive the list of messages as input but will return a corrected version of the message as if corrected by a human
# also to trick the LLM as if the critique is done by a human .. so that there is a simulation of back and forth messages with the LLM
def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflection_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]

# Construct the graph
builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

# Implement the function which should decide whether the response from LLM is good enough or we reflect and continue
def should_continue(state: List[BaseMessage]):
    if len(state) >= 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

# Visualise the graph
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()



if __name__=='__main__':
    print("Hello Langgraph!")
    inputs = HumanMessage(content="""Make this Tweet Better:
                        @LangChainAI
                        - newly Tool Calling feature is seriously underrated.
                        After a long wait , its here making the implementation of agents across different models with function calls.
                        Made a Video covering their newest blog post
                        """)
    response = graph.invoke(inputs)
    print(response)