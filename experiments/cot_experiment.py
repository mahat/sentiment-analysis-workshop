import numpy as np
import random
from experiments.utils import remove_tags
from experiments.simple_prompt_experiment import SimplePromptExperiment

from datasets import load_dataset

class CotExperiment(SimplePromptExperiment):

    system_prompt = """
You are a helpful large language model that understands sentiment of a movie review inside <review>...</review> tags. 
Respond in the following format, using careful step-by-step reasoning.
<reasoning>
...
</reasoning>
<answer>
...
</answer>
Your answer must be either Negative or Positive
""".strip()
    user_prompt = """<review>{review}</review>
""".strip()

    # def __init__(self, repo_id, file_name, n_ctx, valid_ds):
    #     super().__init__(repo_id, file_name, n_ctx, valid_ds)

    # def parse_response(self,response: dict) -> int:
    #     super().parse_response()

if __name__ == '__main__':
    repo_id="bartowski/Qwen2.5-0.5B-Instruct-GGUF"
    file_name="Qwen2.5-0.5B-Instruct-IQ2_M.gguf"
    n_ctx = 4096
    #verbose=False
    ds = load_dataset('ajaykarthick/imdb-movie-reviews')
    valid_ds = ds['train'].shuffle().select(range(0,100))
    # ds_train_subset = ds['train'].shuffle().select(range(0,100))
    exp = CotExperiment(repo_id, file_name, n_ctx=4096, valid_ds=valid_ds)
    print(exp().eval())
    # to prevent error
    del exp.llm
    from IPython import embed; embed()
