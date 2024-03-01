import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from pprint import pprint
from DataWorker import DataWorker

class VectorDataBase:
    def __init__(self, embedding_model,dataset_name,dataset_column):
        self.embedding_model = embedding_model
        self.verbose = False
        self.data_worker = DataWorker(dataset_name, dataset_column)
        self.db = self._GenerateVectorDB()        
        
    def TurnVerboseModeOn(self):
        self.verbose = True
    
    def TurnVerboseModeOff(self):
        self.verbose = False
        
    def _GetDevice(self):
        if torch.cuda.is_available():
            device='cuda'
        else:
            device='cpu'
        print(f"Device Allocated: {device}")
        return device
        
    def _GetEmbeddingModel(self):
        # Create a dictionary with model configuration options, specifying to use the GPU for computations
        model_kwargs = {'device':self._GetDevice()}
        # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
        encode_kwargs = {'normalize_embeddings': False}
        # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,     # Provide the pre-trained model's path
            model_kwargs=model_kwargs, # Pass the model configuration options
            encode_kwargs=encode_kwargs # Pass the encoding options
        )
        return embeddings 
        
    def _GenerateVectorDB(self):
        docs = self.data_worker.GetDocs()
        embeddings = self._GetEmbeddingModel()
        db = FAISS.from_documents(docs, embeddings)
        return db
    
    def GetRetriever(self,query):
        kwargs = {"query":query,
                  "k":2,
                  "fetch_k":5}
        results = self.db.similarity_search(**kwargs)
        if self.verbose:
            for doc in results:
                print("\n----MOST SIMILAR DOCUMENTS----")
                pprint(doc)
        # Create a retriever object from the 'db' using the 'as_retriever' method.
        # This retriever is likely used for retrieving data or documents from the database.
        # a search configuration where it retrieves up to 4 relevant splits/documents.
        retriever = self.db.as_retriever(search_kwargs={"k":4})
        docs = retriever.get_relevant_documents(query)
        if self.verbose:
            for doc in docs:
                print(f"\n----RETRIEVER DOCUMENT----\n{doc}\n")
        return retriever