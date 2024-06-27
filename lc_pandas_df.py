import sys, json
import pandas as pd
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

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
        print("Hello LangChain Pandas Dataframe!")

        model = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, azure_endpoint=azure_apibase, temperature=0)
        embed = AzureOpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, azure_endpoint=azure_apibase)
         
        run_option = 0        
        match run_option:
            case 0:
                df = pd.read_csv("./data/titanic.csv")

                # The ZERO_SHOT_REACT_DESCRIPTION agent
                agent = create_pandas_dataframe_agent(model, df, verbose=True, allow_dangerous_code=True)

                # The OPENAI_FUNCTIONS agent
                agent1 = create_pandas_dataframe_agent(
                    model,
                    df,
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    allow_dangerous_code=True,
                )
                
                print(agent.invoke("how many rows are there?"))
            case _:
                print(f'Error: Wrong run_option({run_option})!')

    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())