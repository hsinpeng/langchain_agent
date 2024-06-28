import sys, json
# Import relevant functionality
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import AzureChatOpenAI

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

def main():
    try:
        print("Hello, LangChain Agent!")

        model = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, azure_endpoint=azure_apibase, temperature=0.9)

        # Create the agent
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        tools = [wikipedia]

        run_option = 1
        match run_option:
            case 0:
                model_with_tools = model.bind_tools(tools)
                response = model_with_tools.invoke([HumanMessage(content="hi im bob! and i live in taoyuan city")])
                print(f"ContentString: {response.content}")
                print(f"ToolCalls: {response.tool_calls}")

                response = model_with_tools.invoke([HumanMessage(content="who is the mayor of taoyuan city?")])
                print(f"ContentString: {response.content}")
                print(f"ToolCalls: {response.tool_calls}")
            case 1:
                memory = SqliteSaver.from_conn_string(":memory:")
                agent_executor = create_react_agent(model, tools, checkpointer=memory)
                # Use the agent
                config = {"configurable": {"thread_id": "abc123"}}
                for chunk in agent_executor.stream(
                    {"messages": [HumanMessage(content="hi im bob! and i live in taoyuan city")]}, config
                ):
                    print(chunk)
                    print("----")
                for chunk in agent_executor.stream(
                    {"messages": [HumanMessage(content="who is the president of the country of where I live?")]}, config
                ):
                    print(chunk)
                    print("----")
            case _:
                print(f'Error: Wrong run_option({run_option})!')

    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())