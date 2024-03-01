from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle 
import os

class DataWorker:
    
    def __init__(self, dataset_name, dataset_column):
       self.dataset_name = dataset_name
       self.dataset_folder = os.path.join(os.getcwd(), 'datasets')
       os.makedirs(self.dataset_folder, exist_ok=True)  # Ensure the datasets folder exists
       self.data = self._SplitText(self._LoadData(dataset_name, dataset_column))
        
    def _LoadData(self, dataset_name, dataset_column):
        pickle_path = os.path.join(self.dataset_folder, f"{dataset_name.replace('/', '_')}_{dataset_column}.pkl")
        
        if os.path.exists(pickle_path):
            return self._LoadPickle(pickle_path)
        else:
            loader = HuggingFaceDatasetLoader(dataset_name, dataset_column)
            dataset = loader.load()
            self._SavePickle(dataset, pickle_path)
            return dataset
        
    def _SplitText(self,dataset):
        # Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
        # It splits text into chunks of 1000 characters each with a 150-character overlap.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        # 'data' holds the text you want to split, split the text into documents using the text splitter.
        docs = text_splitter.split_documents(dataset)
        return docs
    
    def _SavePickle(self, data, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(data, file)
 
    def _LoadPickle(self, filepath):
         with open(filepath, 'rb') as file:
             return pickle.load(file)
    
    def GetDocs(self):
        return self.data