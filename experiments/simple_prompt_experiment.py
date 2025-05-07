#from experiment import Experiment
from experiments.experiment import Experiment
import numpy as np
from datasets import load_from_disk

from experiments.utils import remove_tags,evaluator

class SimplePromptExperiment(Experiment):

    system_prompt = """
You are a helpful large language model that understands sentiment of a review inside <review>...</review> tags. 
Your final answer must be either Negative or Positive based on the sentiment of the review.
""".strip()
    
    user_prompt = """
<review>{review}</review> 

Your final answer is """.strip()

    def template_prompt(self, text:str) -> str:
        # clean html tags in text and replace review tag
        return self.user_prompt.format(review=remove_tags(text))
    
    def parse_response(self,response: dict) -> int:
        try:
            content = response['choices'][0]['message']['content'].lower()
        except Exception:
            print("parse error")
            return -1
        
        if 'positive' in content: # equality check instead
            return 0
        elif 'negative' in content:
            return 1
        else:
            return -1 # missing case 
    
    def eval(self,args=None):
        assert len(self.preds) > 0, "predictions are not found! Call Experiment First"
        # calculate missings 
        preds = np.array(self.preds)
        true = np.array(self.labels)
        valid_answer_ratio = (preds != -1).sum() / len(self.valid_ds)
        eval_dict = evaluator(y_pred=preds[preds != -1],y_true=true[preds != -1])
        # eval_dict = evaluator(y_pred=preds,y_true=true)
        eval_dict['valid_answer_ratio'] = valid_answer_ratio
        return eval_dict
    

# create experiment and run
if __name__ == '__main__':
    # small model
    repo_id="bartowski/Qwen2.5-0.5B-Instruct-GGUF"
    file_name="Qwen2.5-0.5B-Instruct-IQ2_M.gguf"
    # big model
    repo_id="bartowski/Qwen2.5-1.5B-Instruct-GGUF"
    file_name="Qwen2.5-1.5B-Instruct-IQ2_M.gguf"

    n_ctx = 4096
    #verbose=False
    valid_ds = load_from_disk('./data/valid_small.hf')
    # ds_train_subset = ds['train'].shuffle().select(range(0,100))
    exp = SimplePromptExperiment(model=None, repo_id=repo_id, file_name=file_name, n_ctx=4096, valid_ds=valid_ds)
    print(exp().eval())
    # to prevent error
    del exp.llm
    from IPython import embed; embed()