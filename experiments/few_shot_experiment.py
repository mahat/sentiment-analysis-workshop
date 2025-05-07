import numpy as np
import random
from experiments.utils import remove_tags
from experiments.simple_prompt_experiment import SimplePromptExperiment

from datasets import load_dataset,load_from_disk

class FewShotExperiment(SimplePromptExperiment):

    system_prompt = """
You are a helpful large language model that understands sentiment of a movie review. Examples are provided inside <examples>...</examples> tags as list items <li>...</li> to help you to decide.
Your final answer must be either Negative or Positive based on the sentiment of the review.
""".strip()
    user_prompt = """
<examples>
{inContext}
</examples>
Review: {query} Final Answer: """.strip()

    def __init__(self, repo_id, file_name, n_ctx, valid_ds, incoxtext_examples, pass_k=2):
        super().__init__(model=None, repo_id=repo_id, file_name=file_name, n_ctx=n_ctx, valid_ds=valid_ds)
        self.incoxtext_examples = incoxtext_examples
        self.pass_k = pass_k

    def template_prompt(self, text:str) -> str:
        # append in_context
        random_index = random.choices(range(0,len(self.incoxtext_examples)), k=self.pass_k)
        incontext_template = "<li>\nReview: {review} Final Answer: {sentiment}\n</li>\n"
        sentiment_to_str = lambda x: "Positive" if x == 0 else "Negative"
        incontext_str = ""
        for i in random_index:
            incontext_str += incontext_template.format(review=remove_tags(self.incoxtext_examples['review'][i]), sentiment=sentiment_to_str(self.incoxtext_examples['label'][i]))
        # clean html tags in text

        return self.user_prompt.format(inContext=incontext_str, query=remove_tags(text))

if __name__ == '__main__':
    repo_id="bartowski/Qwen2.5-0.5B-Instruct-GGUF"
    file_name="Qwen2.5-0.5B-Instruct-IQ2_M.gguf"
    n_ctx = 4096
    #verbose=False
    ds = load_dataset('ajaykarthick/imdb-movie-reviews')
    valid_ds = ds['train'].shuffle().select(range(0,100))
    incoxtext_examples = ds['train'].shuffle().select(range(0,32))
    # ds_train_subset = ds['train'].shuffle().select(range(0,100))
    exp = FewShotExperiment(repo_id, file_name, n_ctx=1024*16, valid_ds=valid_ds,incoxtext_examples=incoxtext_examples, pass_k=32)
    print(exp().eval())
    # to prevent error
    del exp.llm
    from IPython import embed; embed()