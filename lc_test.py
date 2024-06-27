import sys, json
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

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
        print("Hello LangChain!")

        model = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, azure_endpoint=azure_apibase, temperature=0.9)
        embed = AzureOpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, azure_endpoint=azure_apibase)
         
        run_option = 0        
        match run_option:
            case 0:
                print('Hello')
            case _:
                print(f'Error: Wrong run_option({run_option})!')

    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())