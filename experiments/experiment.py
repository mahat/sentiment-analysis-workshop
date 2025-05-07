from abc import ABC, abstractmethod
from typing import Any,Iterable
from llama_cpp import Llama
from tqdm import tqdm

class Experiment(ABC):
    
    # system_prompt = None
    # user_prompt = None
    llm = None
    preds = []
    labels = []
    valid_ds = None

    def __init__(self,model: Llama=None, repo_id:str="", file_name:str="", n_ctx:int=4096, valid_ds:Iterable[Any]=[]):
        self.repo_id = repo_id
        self.file_name = file_name
        self.n_ctx = n_ctx
        self.valid_ds = valid_ds
        self.stop = []
        self.max_tokens = 256
        self.repeat_penalty = 1.0
        self.temperature = 0.2
        self.min_p = 0.05
        # other parameters
        self.run_log = []
        self.top_k = 40
        self.logit_bias = None
        self.llm = model 

        if model is None:
            self.llm = Llama.from_pretrained(
                    repo_id=self.repo_id,
                    filename=self.file_name,
                    n_ctx = n_ctx,
                    verbose=False,
                    seed = 42,
                )


    @abstractmethod
    def template_prompt(self, text:str) -> str:
        # converts user context 
        pass
    
    @abstractmethod
    def parse_response(self,response: dict) -> int:
        # 0 -> positive, 1 -> negative
        pass

    def model_call(self, user_prompt):
        # reset context for unbiased generation
        self.llm.reset()
        # generate 
        response = self.llm.create_chat_completion(
            messages = [
                {   
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            seed=42,
            temperature=self.temperature,
            repeat_penalty=self.repeat_penalty,
            max_tokens=self.max_tokens,  # Adjust as needed
            min_p=self.min_p,
            top_k=self.top_k,
            stop=self.stop, #Add stop words,
            logit_bias=self.logit_bias
            # **params
        )
        return response

    def __call__(self, generation_params={}):
        # flush preds and labels to prevent accumulation
        self.preds = []
        self.labels = []
        # runs an prompts on valid_ds and stores predictions and labels 
        for e in tqdm(self.valid_ds,desc="running validation set"):
            user_prompt = self.template_prompt(e['review'])
            response = self.model_call(user_prompt=user_prompt,**generation_params)
            self.labels.append(e['label'])
            self.preds.append(self.parse_response(response))
            self.run_log.append({'system_promt':self.system_prompt,'user_prompt':user_prompt, 'pred':self.preds[-1], 'label':self.labels[-1],'response':response})
        return self
    
    def get_run_log(self):
        return self.run_log
    
    def single_result_generation(self,text):
        user_prompt = self.template_prompt(text)
        response = self.model_call(user_prompt=user_prompt)
        return self.parse_response(response)
    
    @abstractmethod
    def eval(self,arg=None):
        # assert preds or labels, "preds and labels are empty please run an experiment"
        # # calculate evaluation accuracy
        pass
    
    