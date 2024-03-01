import chainlit as cl
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from utils.VectorDataBase import VectorDataBase
from utils.Evaluator import Evaluator
from utils.LLM import LLM
import warnings
from chainlit.input_widget import Select
warnings.simplefilter('ignore')   

"""
vec_kwargs = {
    'dataset_name':"databricks/databricks-dolly-15k",
    'dataset_column': "context",
    'embedding_model':'sentence-transformers/sentence-t5-base'}
llm_kwargs = {
    'llm_model': "gpt-3.5-turbo"}


vdb = VectorDataBase(**vec_kwargs )
llm = LLM(vdb,**llm_kwargs)
"""

@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            Select(
                id="dataset",
                label="Dataset for VectorDatabase",
                values=["databricks/databricks-dolly-15k"],
                initial_index=0,
            ),
            Select(
                id="column",
                label="Column from Dataset",
                values=["context",'instruction','response'],
                initial_index=0,
            ),
            Select(
                id="llm model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
                initial_index=0,
            ),
            Select(
                id="embedding model",
                label="Embbeding - Model",
                values=['sentence-transformers/sentence-t5-base'],
                initial_index=0,
            )
        ]
    ).send()
    await setup_agent(settings)

@cl.on_settings_update
async def setup_agent(settings):
    print("Setup agent with following settings: ", settings)
    
    vec_kwargs = {
        'dataset_name':settings['dataset'],
        'dataset_column': settings["column"],
        'embedding_model':settings["embedding model"]}
    llm_kwargs = {
        'llm_model': settings["llm model"]}
    
    
    vdb = VectorDataBase(**vec_kwargs )
    llm = LLM(vdb, **llm_kwargs)

    cl.user_session.set("llm", llm)


@cl.on_message
async def on_message(msg: cl.Message):
    llm = cl.user_session.get("llm")
    res = await cl.AskActionMessage(
        content="Pick an action!",
        actions=[
           cl.Action(name="Use RAG System", value="True", description="Load VDB for additional context"),
            cl.Action(name="Use LLM (No RAG)", value="False", label="Let the LLM perform based on the initial training"),
        ],
    ).send()
    
    if res and res.get("value") == "True":
        response = llm.GetResponse(msg.content, use_vdb=True)
    elif res and res.get("value") == "False":
        response = llm.GetResponse(msg.content, use_vdb=False)
    else:
        response = "Error Code: 1 - Neither VDB or LLM was selected"
    await cl.Message(content=response).send()

@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")

@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")