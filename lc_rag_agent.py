import sys, json
import bs4
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver

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
        print("Hello LangChain RAG Agent!")

        model = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, azure_endpoint=azure_apibase, temperature=0.9)
        embed = AzureOpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, azure_endpoint=azure_apibase)
        
        # Load, chunk and index the contents of the blog.
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
                
        # Retrieve and generate using the relevant snippets of the blog.
        vectorstore = Chroma.from_documents(documents=splits, embedding=embed)                
        retriever = vectorstore.as_retriever()

        # Retrieval tool
        tool = create_retriever_tool(
            retriever,
            "blog_post_retriever",
            "Searches and returns excerpts from the Autonomous Agents blog post.",
        )
        tools = [tool]
        #print(tool.invoke("task decomposition"))

        run_option = 1      
        match run_option:
            case 0:
                # Agent constructor
                agent_executor = create_react_agent(model, tools)

                query = "What is Task Decomposition?"

                for s in agent_executor.stream(
                    {"messages": [HumanMessage(content=query)]},
                ):
                    print(s)
                    print("----")

            case 1:
                # Agent constructor with memory
                memory = SqliteSaver.from_conn_string(":memory:")

                agent_executor = create_react_agent(model, tools, checkpointer=memory)

                config = {"configurable": {"thread_id": "abc123"}}

                query = "Hi! I'm bob"
                for s in agent_executor.stream(
                    {"messages": [HumanMessage(content=query)]}, config=config
                ):
                    print(s)
                    print("----")

                query = "What is Task Decomposition?"
                for s in agent_executor.stream(
                    {"messages": [HumanMessage(content=query)]}, config=config
                ):
                    print(s)
                    print("----")

                query = "What according to the blog post are common ways of doing it? redo the search"
                for s in agent_executor.stream(
                    {"messages": [HumanMessage(content=query)]}, config=config
                ):
                    print(s)
                    print("----")

            case _:
                print(f'Error: Wrong run_option({run_option})!')

    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())