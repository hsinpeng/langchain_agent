# langchain_agent
My Test of LangChain Agent/Tools

# Introduction 
My personal study of LangChain Agent/Tools with Azure OpenAI. The most content are modified from the official langchain document.

The official langchain document website: https://python.langchain.com/docs/

# Getting Started
1. Clone this project from GitHub.
```text=
git clone [Project URL]
```

2. Enter project folder (langchain_examples).
```text=
cd langchain_agent
```

3. New the parameter file (param.json) in the project folder, the formate is as the following.

Note: Before using this project, you have to set and get these parameter from Azure AI service.
```text=
{
    "hostname": "Please, do it by yourself",
    "hostport": "Please, do it by yourself",
    "azure_apikey" : "Please, do it by yourself", 
    "azure_apibase"  : "Please, do it by yourself",
    "azure_apitype" : "azure",
    "azure_apiversion" : "Please, do it by yourself",
    "azure_gptx_deployment" : "Please, do it by yourself",
    "azure_embd_deployment" : "Please, do it by yourself",
}
```

4. Install related python packages.

5. The main program is 'agent.py', check it and learn how to use it by inspecting source code.

6. Program execution.
```text=
python agent.py
```

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 
