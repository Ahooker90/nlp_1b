from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering,pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from pprint import pprint
import torch
import warnings
import re
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
client = openai.OpenAI()

print_citations = False

def get_completion(prompt, model="gpt-3.5-turbo"):

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


def PrintCitations():
    cite_list = [
        "https://medium.com/international-school-of-ai-data-science/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7",
        "https://python.langchain.com/docs/integrations/vectorstores/faiss",
        "https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.html",
        ]
    print("\nThis work drew inspiration from previous work. Here are a few of the primary sources used.")
    pprint(cite_list)

def LoadData(dataset_name="databricks/databricks-dolly-15k", dataset_column="context"):
    # Specify the dataset name and the column containing the content
    # Create a loader instance
    loader = HuggingFaceDatasetLoader(dataset_name, dataset_column)
    # Load the data
    dataset = loader.load()
    return dataset
    
def SplitText(dataset):
    # Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
    # It splits text into chunks of 1000 characters each with a 150-character overlap.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    # 'data' holds the text you want to split, split the text into documents using the text splitter.
    docs = text_splitter.split_documents(dataset)
    return docs

def GetDevice():
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    print(f"Device Allocated: {device}")
    return device

def GetEmbeddingModel(modelPath):
    # Create a dictionary with model configuration options, specifying to use the GPU for computations
    model_kwargs = {'device':GetDevice()}
    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}
    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )
    return embeddings 

def GenerateVectorDB(embedding_model):
    dataset = LoadData()
    docs =SplitText(dataset)
    embeddings = GetEmbeddingModel(embedding_model)
    db = FAISS.from_documents(docs, embeddings)
    return db

def GetRetriever(db, query, verbose=True):
    kwargs = {"query":query,
              "k":2,
              "fetch_k":5}
    results = db.similarity_search(**kwargs)
    if verbose:
        for doc in results:
            print("\n----MOST SIMILAR DOCUMENTS----")
            pprint(doc)
    # Create a retriever object from the 'db' using the 'as_retriever' method.
    # This retriever is likely used for retrieving data or documents from the database.
    # a search configuration where it retrieves up to 4 relevant splits/documents.
    retriever = db.as_retriever(search_kwargs={"k":4})
    docs = retriever.get_relevant_documents(query)
    if verbose:
        for doc in docs:
            print(f"\n----RETRIEVER DOCUMENT----\n{doc}\n")
    return retriever

def LoadLLMPipeline():
    # Specify the model name you want to use
    model_name = "distilbert/distilbert-base-cased-distilled-squad"
    
    # Load the tokenizer associated with the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)
    
    # Define a question-answering pipeline using the model and tokenizer
    question_answerer = pipeline(
        "question-answering", 
        model=model_name,
        temperature=0.7,
        tokenizer=tokenizer,
        return_tensors='pt'
    )

    return question_answerer

def GetPrompt(context, query):
    prompt = f"""
    You are a helpful assistnat that answers questions. You are not very good at
    remembering information so any time you see information denoted in <<>> use this 
    to answer questions.It is important to not guess. If you do not know, simply say so.
    
    you will have any information being provided to you clearly shown in the following format:
    
    Information: <<{context}>>
    Question:'''{query}'''
    Answer:
    
    For example: 
    Information: <<The sky is blue>>
    Question:'''What color is the sky'''
    Answer: Blue
    
    For example: 
    Information: <<Cars have four wheels>>
    Question:'''How many wheels do motorcycles have?'''
    Answer: I Dont Know
    
    For example: 
    Information: <<>>
    Question:'''How many wheels do motorcycles have?'''
    Answer: Two
    
    """
    return prompt


def GetAnswer(llm_output):
    match = re.search(r'Answer:\s*(.*)', llm_output)
    # Extract the answer if it was found
    answer = match.group(1) if match else 'Answer not found'
    return answer




if __name__ == '__main__':
    warnings.simplefilter('ignore')
    model = 'sentence-transformers/sentence-t5-base'
    
    query = "Who is the current president of USA?"
    db = GenerateVectorDB(model)
    retriever = GetRetriever(db, query)
    
    context=retriever.get_relevant_documents(query)
    prompt =GetPrompt(context, query)

    out_0 = GetAnswer(get_completion(prompt))
    prompt =GetPrompt(context, "")
    out_1 = GetAnswer(get_completion(prompt))
    
    print(f"\nWith Context: {out_0}\n")
    print(f"Without Context: {out_1}\n")
    
    if print_citations:
        PrintCitations()















