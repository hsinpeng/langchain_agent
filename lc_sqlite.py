##### Build a Question/Answering system over SQL data #####
import sys, json
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


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
        print("Hello LangChain SQL query!")

        model = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, azure_endpoint=azure_apibase, temperature=0.9)
        db = SQLDatabase.from_uri("sqlite:///data/Chinook.db")

        run_option = 2      
        match run_option:
            case 0:
                chain = create_sql_query_chain(model, db)
                response = chain.invoke({"question": "How many employees are there"})
                print(response)
                print(db.run(response))
                print(chain.get_prompts()[0].pretty_print())

            case 1:
                write_query = create_sql_query_chain(model, db)
                execute_query = QuerySQLDataBaseTool(db=db)
                chain = write_query | execute_query
                response = chain.invoke({"question": "How many employees are there"})
                print(response)
            case 2:
                answer_prompt = PromptTemplate.from_template(
                    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

                Question: {question}
                SQL Query: {query}
                SQL Result: {result}
                Answer: """
                )
                write_query = create_sql_query_chain(model, db)
                execute_query = QuerySQLDataBaseTool(db=db)
                chain = (
                    RunnablePassthrough.assign(query=write_query).assign(
                        result=itemgetter("query") | execute_query
                    )
                    | answer_prompt
                    | model
                    | StrOutputParser()
                )

                response = chain.invoke({"question": "How many employees are there"})
                print(response)
            case _:
                print(f'Error: Wrong run_option({run_option})!')

    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())