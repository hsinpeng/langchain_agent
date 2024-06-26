import sys, json
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import AzureChatOpenAI #ChatOpenAI
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

###### Azure OpenAI Settings #######
with open('param.json', 'r', encoding='utf-8') as param_file:
    param_data = json.load(param_file)
    azure_apikey = param_data["azure_apikey"]
    azure_apibase  = param_data["azure_apibase"]
    azure_apitype = param_data["azure_apitype"]
    azure_apiversion = param_data["azure_apiversion"]
    azure_gptx_deployment = param_data["azure_gptx_deployment"]
    azure_embd_deployment = param_data["azure_embd_deployment"]
param_file.close()
####################################

# Define the tools for the agent to use
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [wikipedia]
tool_node = ToolNode(tools)

model = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, azure_endpoint=azure_apibase, temperature=0.9)
model.bind_tools(tools)

# Define the function that determines whether to continue or not
def should_continue(state: AgentState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END

# Define the function that calls the model
def call_model(state: AgentState):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

def main():
    try:
        print("Hello Langgraph!")
        
        run_option = 0
        match run_option:
            case 0:
                # Define a new graph
                workflow = StateGraph(MessagesState)

                # Define the two nodes we will cycle between
                workflow.add_node("agent", call_model)
                workflow.add_node("tools", tool_node)

                # Set the entrypoint as `agent`
                # This means that this node is the first one called
                workflow.set_entry_point("agent")

                # We now add a conditional edge
                workflow.add_conditional_edges(
                    # First, we define the start node. We use `agent`.
                    # This means these are the edges taken after the `agent` node is called.
                    "agent",
                    # Next, we pass in the function that will determine which node is called next.
                    should_continue,
                )

                # We now add a normal edge from `tools` to `agent`.
                # This means that after `tools` is called, `agent` node is called next.
                workflow.add_edge("tools", 'agent')

                # Initialize memory to persist state between graph runs
                checkpointer = MemorySaver()

                # Finally, we compile it!
                # This compiles it into a LangChain Runnable,
                # meaning you can use it as you would any other runnable.
                # Note that we're (optionally) passing the memory when compiling the graph
                app = workflow.compile(checkpointer=checkpointer)

                # Use the Runnable
                final_state = app.invoke(
                    {"messages": [HumanMessage(content="who is the president of taiwan?")]},
                    config={"configurable": {"thread_id": 42}}
                )
                print(final_state["messages"][-1].content)
            case _:
                print(f'Error: Wrong run_option({run_option})!')

    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())