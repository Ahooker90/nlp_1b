import openai
from dotenv import load_dotenv, find_dotenv
import re
#model="gpt-3.5-turbo"
class LLM:
    
    def __init__(self,vdb,llm_model):
        self.client = self._CreateClient()
        self.model = llm_model
        self.vdb=vdb
        self.use_vdb = False
    
    def SetUseVdbOn(self):
        self.use_vdb = True
    
    def SetUseVdbOff(self):
        self.use_vdb = False
        
    def _CreateClient(self):
        _ = load_dotenv(find_dotenv()) # read local .env file
        client = openai.OpenAI()
        return client
    
    def _GetCompletion(self,prompt):
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
        model=self.model,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content

    def _GetAnswerPrompt(self,context, query):
        prompt = f"""
        You are a helpful assistnat that answers questions. You are not very good at
        remembering information so any time you see information denoted in <<>> use this 
        to answer questions.It is important to not guess. If you do not know, simply say so.
        
        you will have any information being provided to you clearly shown in the following format:
        
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
        
        Now Answer the follwing Question:
            Information: <<{context}>>
            Question:'''{query}'''

        """
        return prompt

    def _GetReponsePrompt(self,context,query):
        prompt = f"""
        If given context it will be clearly denoted in between triple back ticks '''
        Use this to answer question, if it is available.
        
        Context: '''{context}'''
        
        Answer the following question with context if context is available:
            
        Question: {query}
        """
        return prompt


    def _GetContextFromVdb(self,query):
        retriever = self.vdb.GetRetriever(query)
        context=retriever.get_relevant_documents(query)
        return context

    def GetAnswer(self,query,use_vdb=False,verbose=False):
        context = ""
        if use_vdb:
            context = self._GetContextFromVdb(query)

        prompt=self._GetAnswerPrompt(context, query)
        llm_output = self._GetCompletion(prompt)
        
        if verbose:
            print(llm_output)
        
        match = re.search(r'Answer:\s*(.*)', llm_output)
        # Extract the answer if it was found
        answer = match.group(1) if match else 'Answer not found'

        return answer

    def GetResponse(self, query, use_vdb = False):
        context = ""
        if use_vdb:
            context = self._GetContextFromVdb(query)

        prompt=self._GetReponsePrompt(context, query)
        
        return self._GetCompletion(prompt)
    












