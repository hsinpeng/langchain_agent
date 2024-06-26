import sys, json
from langchain_community.utilities import SQLDatabase

#import sqlite3
#con = sqlite3.connect("./data/Chinook.db")
#cur = con.cursor()
#res = cur.execute("SELECT * FROM Artist LIMIT 10;")
#print(res.fetchall())

db = SQLDatabase.from_uri("sqlite:///data/Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())
print(db.run("SELECT * FROM Artist LIMIT 10;"))

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
