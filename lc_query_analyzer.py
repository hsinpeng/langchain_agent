# Build a Query Analysis System
import sys, json, datetime
from typing import Optional, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

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

# Query schema: Explicit min and max attributes for publication date so that it can be filtered on
class Search(BaseModel):
    # """Search over a database of tutorial videos about a software library."""
    query: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    publish_year: Optional[int] = Field(None, description="Year video was published")


def main():
    try:
        print("Hello, LangChain Query Analyzer!")

        model = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, azure_endpoint=azure_apibase, temperature=0)
        embed = AzureOpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, azure_endpoint=azure_apibase)

        # Use the YouTubeLoader to load transcripts of a few LangChain videos
        urls = [
            "https://www.youtube.com/watch?v=HAn9vnJy6S4",
             "https://www.youtube.com/watch?v=dA1cHGACXCo",
            "https://www.youtube.com/watch?v=ZcEMLz27sL4",
            "https://www.youtube.com/watch?v=hvAPnpSfSGo",
            "https://www.youtube.com/watch?v=EhlPDL4QrWY",
            "https://www.youtube.com/watch?v=mmBo8nlu2j0",
            "https://www.youtube.com/watch?v=rQdibOsL1ps",
            "https://www.youtube.com/watch?v=28lC4fqukoc",
            "https://www.youtube.com/watch?v=es-9MgxB-uc",
            "https://www.youtube.com/watch?v=wLRHwKuKvOE",
            "https://www.youtube.com/watch?v=ObIltMaRJvY",
            "https://www.youtube.com/watch?v=DjuXACWYkkU",
            "https://www.youtube.com/watch?v=o7C9ld6Ln-M",
        ]
        docs = []
        for url in urls:
            transcript = YoutubeLoader.from_youtube_url(url, add_video_info=True).load()
            #print(transcript)
            docs.extend(transcript)

        # Add some additional metadata: what year the video was published
        for doc in docs:
            doc.metadata["publish_year"] = int(datetime.datetime.strptime(doc.metadata["publish_date"], "%Y-%m-%d %H:%M:%S").strftime("%Y"))

        #print([doc.metadata["title"] for doc in docs])
        #for doc in docs:
        #    print(doc.metadata)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        chunked_docs = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            chunked_docs,
            embed,
        )     

         
        run_option = 1      
        match run_option:
            case 0:
                # Retrieval without query analysis
                search_results = vectorstore.similarity_search("how do I build a RAG agent")
                print(search_results[0].metadata["title"])
                print(search_results[0].page_content[:500])
                
                # Retrieval without query analysis: Search for results from a specific time period?
                search_results = vectorstore.similarity_search("videos on RAG published in 2023")
                print(search_results[0].metadata["title"])
                print(search_results[0].metadata["publish_date"]) # The result is "No Good"!
                print(search_results[0].page_content[:500])

            case 1:
                # Query generation: Convert user questions to structured queries we'll make use of OpenAI's tool-calling API.
                system = """You are an expert at converting user questions into database queries. \
                You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
                Given a question, return a list of database queries optimized to retrieve the most relevant results.

                If there are acronyms or words you are not familiar with, do not try to rephrase them."""
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system),
                        ("human", "{question}"),
                    ]
                )

                structured_llm = model.with_structured_output(Search)
                query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm
                print(query_analyzer.invoke("how do I build a RAG agent"))
                print(query_analyzer.invoke("videos on RAG published in 2023"))

                # Retrieval with query analysis
                def retrieval(search: Search) -> List[Document]:
                    if search.publish_year is not None:
                        # This is syntax specific to Chroma, the vector database we are using.
                        _filter = {"publish_year": {"$eq": search.publish_year}}
                    else:
                        _filter = None
                    return vectorstore.similarity_search(search.query, filter=_filter)
                
                retrieval_chain = query_analyzer | retrieval

                results = retrieval_chain.invoke("RAG tutorial published in 2023")
                [(doc.metadata["title"], doc.metadata["publish_date"]) for doc in results]
                #print(results[0].metadata["title"])
                #print(results[0].metadata["publish_date"]) # The result is "No Good"!
                #print(results[0].page_content[:500])

            case _:
                print(f'Error: Wrong run_option({run_option})!')

    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())