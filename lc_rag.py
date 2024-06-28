# Build a Retrieval Augmented Generation (RAG) App
import sys, json
import bs4
from langchain import hub
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

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
store = {}

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def main():
    try:
        print("Hello, LangChain RAG!")

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
        #print(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        #print(f'splits count = {len(splits)}')
        #for chun in splits:
        #    print(f'page_content\n{chun.page_content}')
        #    print(f'metadata\n{chun.metadata}')
                
        # Retrieve and generate using the relevant snippets of the blog.
        vectorstore = Chroma.from_documents(documents=splits, embedding=embed)                
        retriever = vectorstore.as_retriever()

        run_option = 4  
        match run_option:
            case 0:
                # Prompt Template
                #prompt = hub.pull("rlm/rag-prompt")
                human_prompt="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
                prompt = ChatPromptTemplate.from_messages([("user", human_prompt)])

                # LLM chain and invoke
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | model
                    | StrOutputParser()
                )

                for chunk in rag_chain.stream("What is Task Decomposition?"):
                    print(chunk, end="", flush=True)
                print("\n")
            
            case 1:
                # Built-in chains
                system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Use three sentences maximum and keep the "
                    "answer concise."
                    "\n\n"
                    "{context}"
                )
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ]
                )

                question_answer_chain = create_stuff_documents_chain(model, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                response = rag_chain.invoke({"input": "What is Task Decomposition?"})
                print(response["answer"])
            
            case 2:
                # Customizing the prompt
                template = """Use the following pieces of context to answer the question at the end.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Use three sentences maximum and keep the answer as concise as possible.
                Always say "thanks for asking!" at the end of the answer.

                {context}

                Question: {question}

                Helpful Answer:"""

                custom_rag_prompt = PromptTemplate.from_template(template)

                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | custom_rag_prompt
                    | model
                    | StrOutputParser()
                )

                response = rag_chain.invoke("What is Task Decomposition?")
                print(response)

            case 3:
                # Adding chat history
                contextualize_q_system_prompt = (
                    "Given a chat history and the latest user question "
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do NOT answer the question, "
                    "just reformulate it if needed and otherwise return it as is."
                )

                contextualize_q_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
                history_aware_retriever = create_history_aware_retriever(
                    model, retriever, contextualize_q_prompt
                )

                system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Use three sentences maximum and keep the "
                    "answer concise."
                    "\n\n"
                    "{context}"
                )
                qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
                question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                # Test
                chat_history = []
                question = "What is Task Decomposition?"
                ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
                chat_history.extend(
                    [
                        HumanMessage(content=question),
                        AIMessage(content=ai_msg_1["answer"]),
                    ]
                )

                second_question = "What are common ways of doing it?"
                ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

                print(ai_msg_2["answer"])

            case 4:
                contextualize_q_system_prompt = (
                    "Given a chat history and the latest user question "
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do NOT answer the question, "
                    "just reformulate it if needed and otherwise return it as is."
                )

                contextualize_q_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
                history_aware_retriever = create_history_aware_retriever(
                    model, retriever, contextualize_q_prompt
                )

                system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Use three sentences maximum and keep the "
                    "answer concise."
                    "\n\n"
                    "{context}"
                )
                qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
                question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                # Stateful management of chat history
                conversational_rag_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )

                resp = conversational_rag_chain.invoke(
                    {"input": "What is Task Decomposition?"},
                    config={
                        "configurable": {"session_id": "abc123"}
                    },  # constructs a key "abc123" in `store`.
                )["answer"]
                print(resp)

                resp = conversational_rag_chain.invoke(
                    {"input": "What are common ways of doing it?"},
                    config={"configurable": {"session_id": "abc123"}},
                )["answer"]
                print(resp)

                for message in store["abc123"].messages:
                    if isinstance(message, AIMessage):
                        prefix = "AI"
                    else:
                        prefix = "User"

                    print(f"{prefix}: {message.content}\n")

            case _:
                print(f'Error: Wrong run_option({run_option})!')

    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())