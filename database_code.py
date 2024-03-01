from utils.VectorDataBase import VectorDataBase
from utils.Evaluator import Evaluator
from utils.LLM import LLM
from pprint import pprint
import warnings

warnings.simplefilter('ignore')     
print_citations = False

def PrintCitations():
    cite_list = [
        "https://medium.com/international-school-of-ai-data-science/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7",
        "https://python.langchain.com/docs/integrations/vectorstores/faiss",
        "https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.html",
        ]
    print("\nThis work drew inspiration from previous work. Here are a few of the primary sources used.")
    pprint(cite_list)

if __name__ == '__main__':
    
    vec_kwargs = {
        'dataset_name':"databricks/databricks-dolly-15k",
        'dataset_column': "context",
        'embedding_model':'sentence-transformers/sentence-t5-base'}
    llm_kwargs = {
        'llm_model': "gpt-3.5-turbo"}
     
    query = "Who is the current president of USA?"
    
    vdb=VectorDataBase(**vec_kwargs )
    llm = LLM(vdb,**llm_kwargs)
    
    evaluator = Evaluator(llm, use_vdb=True)
    evaluator.PerformEval(1)
    
    if print_citations:
        PrintCitations()































































