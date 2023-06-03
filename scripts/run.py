import json
from typing import Dict, List, Tuple

import numpy as np
import pysnooper
import torch
from matplotlib import pyplot as plt
from transformers import (DistilBertForSequenceClassification,
                          DistilBertTokenizer, pipeline)

# Environment
GPU_ID = 0
device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
# "facebook/opt-125m"
GENERATOR_NAME = "gpt2"
SENTIMENT_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

tokenizer = DistilBertTokenizer.from_pretrained(SENTIMENT_NAME)
generator = pipeline("text-generation",
    model=GENERATOR_NAME, device=GPU_ID
)
generator.model.config.pad_token_id = generator.tokenizer.eos_token_id
classifier = DistilBertForSequenceClassification.from_pretrained(
    SENTIMENT_NAME, # num_labels=2, id2label=id2label, label2id=label2id
).to(device)

def same_seeds(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
# @pysnooper.snoop()
def evaluate_sentiment(text: List[Dict], **kwargs) -> List:
    """
    Reference:
    - https://huggingface.co/docs/transformers/tasks/sequence_classification
    """

    # data preparation
    text = [obj['generated_text'] for obj in text]

    # sentiment analysis
    inputs = tokenizer(text, return_tensors='pt', padding=True).to(device)
    logits = classifier(**inputs).logits

    # convert to probability
    prob = torch.softmax(logits, dim=1).cpu()

    # hardcode to positive sentiment
    prob = prob[:, 1]
    prob = prob.numpy().tolist()

    return prob

def adversarially_generate_sentences(user_prompt: str, prepend: str = None, strategy: Dict = None) -> List[Dict]:
    """
    Generate candidate sentences given a prompt.

    Arguments
    ---------
    user_prompt : str
        User defined prompt, expect model to continue the sentence.

    prepend : str
        Prepend prompt to the user defined prompt, expect model to do some adversarial behavior.

    strategy : Dict
        Generating strategy.
    """

    if prepend is not None:
        prepend = prepend.strip()

    prompt = user_prompt if prepend is None else f'{prepend} {user_prompt}'.strip()
    print(f'{prompt=}')

    same_seeds(0)
    outputs = generator(prompt, **strategy,)

    # Hide prepending prompt from the generated text, such that the victim user will not
    # notice we are controlling the model.
    if prepend is not None:
        outputs = [{
            'generated_text': obj['generated_text'].replace(prepend, '').strip()
        } for obj in outputs]

    return outputs

@torch.no_grad()
def baseline(prompt: str):
    return evaluate_sentiment([{'generated_text': prompt}])

@torch.no_grad()
def evaluate_strategy(prompt: str, prepend: str, strategy: Dict) -> Tuple[List, List]:
    # -------------------------------------------------------------------------------------------- #

    # plt.figure(figsize=(8, 8))

    # -------------------------------------------------------------------------------------------- #

    # Generate candidates sentences for user given prompts, then evaluate the sentiment of
    # each candidate sentence.
    candidates = adversarially_generate_sentences(
        prompt, prepend=None, strategy=strategy
    )

    prob = evaluate_sentiment(candidates)
    # plt.hist(prob, bins=100, alpha=0.5, label='benign')

    benign = (candidates, prob)

    # -------------------------------------------------------------------------------------------- #

    if prepend == "":
        raise ValueError

    # Generate candidates sentences with prepend prompts, then evaluate the sentiment of
    # each candidate sentence.
    candidates = adversarially_generate_sentences(
        prompt, prepend=prepend, strategy=strategy
    )

    prob = evaluate_sentiment(candidates)
    # plt.hist(prob, bins=100, alpha=0.5, label='adversarial')

    adversarial = (candidates, prob)

    # -------------------------------------------------------------------------------------------- #

    return benign, adversarial

@torch.no_grad()
def evaluate():
    def emit(candidates: List[Dict], prob: List[float]) -> List[Dict]:
        return [
            { 'generated_text': c['generated_text'], 'prob': p }
                for c, p in zip(candidates, prob)
        ]

    def savefig(benign: List, adversarial: List, title: str):
        plt.figure(figsize=(8, 8))

        vec = [benign[1], adversarial[1], ]
        plt.hist(vec, bins=20, alpha=0.5, label=['benign', 'adversarial'])

        plt.legend()
        plt.title(title)
        plt.xlim(0, 1)
        plt.savefig(f'{title}.png')
        plt.clf()

    # User defined prompt.
    prompt = "Teens on social media is a problem many parents and guardians have lost sleep over,"
    prepend = " in at at at"

    # LLM generating strategy.
    beam_search = {
        'do_sample': False, 'early_stopping': True, 'max_length': 50 + 4 + len(prompt.split()),
        'num_beams': 5, 'num_return_sequences': 5, "no_repeat_ngram_size": 2,
    }

    # LLM generating strategy.
    nucleus_sampling = {
        'do_sample': True, 'early_stopping': True, 'max_length': 50 + 4 + len(prompt.split()),
        'top_p': 0.9, 'num_return_sequences': 1000,
    }

    # Evaluate the baseline.
    prob = baseline(prompt)
    print(f'{prob=}')

    # Evaluate the beam-search strategy.
    benign, adversarial = evaluate_strategy(prompt, prepend, beam_search)
    savefig(benign, adversarial, 'beam_search')

    # Store the result to json file.
    beam_search = {
        'param': beam_search, 'benign': emit(*benign), 'adversarial': emit(*adversarial),
    }

    # # Evaluate the nucleus-sampling strategy.
    # benign, adversarial = evaluate_strategy(prompt, prepend, nucleus_sampling)
    # savefig(benign, adversarial, 'nucleus_sampling')

    # # Store the result to json file.
    # nucleus_sampling = {
    #     'param': nucleus_sampling, 'benign': emit(*benign), 'adversarial': emit(*adversarial),
    # }

    # Store the result to json file.
    with open('result.json', 'w') as f:
        json.dump({
            'prompt': prompt,
            'prepend': prepend,
            'strategy': {
                'beam_search': beam_search,
                # 'nucleus_sampling': nucleus_sampling,
            },
        }, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    evaluate()
