from utils.DataWorker import DataWorker
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge 
from bert_score import score


class Evaluator:

    def __init__(self, llm,use_vdb):
        self.llm = llm
        self.use_vdb = use_vdb
        
    def _SentenceBLEU(self,str_0, str_1, verbose=False):
        points = sentence_bleu(str_0, str_1)
        if verbose:
            print(f"BLEU score: {points}")
        return points
    
    def _RogueScore(self,str_0, str_1, verbose=False):
        rouge = Rouge()
        scores= rouge.get_scores(str_0, str_1)[0]['rouge-l']
        if verbose:
            print(f"ROUGE scores: {scores}")
        return scores
    
    def _BERTScore(self,str_0, str_1, verbose=False):
        P, R, F1 = score([str_0], [str_1], lang='en', verbose=False)
        if verbose:
            print(f"BERTScore: Precision: {P.mean()}, Recall: {R.mean()}, F1: {F1.mean()}")
        return P,R,F1
    
    def _GetInstructionList(self):
        data_kwargs ={
            'dataset_name':"databricks/databricks-dolly-15k",
            'dataset_column': "instruction"
            }
        
        instruct_docs = DataWorker(**data_kwargs).GetDocs()
        return instruct_docs
    
    def _GetResponseList(self):
        data_kwargs ={
            'dataset_name':"databricks/databricks-dolly-15k",
            'dataset_column': "response"
            }
        
        response_docs = DataWorker(**data_kwargs).GetDocs()
        return response_docs
    
    def PerformEval(self,n=1):
        instruct_docs = self._GetInstructionList()
        response_docs = self._GetResponseList()
        verbose = True
        for i in range(n):
            str_0 = self.llm.GetResponse(instruct_docs[i].page_content, use_vdb = self.use_vdb)
            str_1 = response_docs[i].page_content
            self._SentenceBLEU(str_0, str_1,verbose)
            self._RogueScore(str_0, str_1,verbose)
            self._BERTScore(str_0, str_1,verbose)
            
            