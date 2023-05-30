import json
from collections import Counter
from typing import Tuple, Dict, List

import numpy as np
import pysnooper
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, pipeline)
from matplotlib import pyplot as plt

# Environment
GPU_ID = 0
device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
# "facebook/opt-125m"
GENERATOR_NAME = "gpt2"
SENTIMENT_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_NAME)
generator = pipeline("text-generation",
    model=GENERATOR_NAME, device=GPU_ID
)
generator.model.config.pad_token_id = generator.tokenizer.eos_token_id
classifier = AutoModelForSequenceClassification.from_pretrained(
    SENTIMENT_NAME, num_labels=2, id2label=id2label, label2id=label2id
).to(device)

def same_seeds(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# @pysnooper.snoop()
@torch.no_grad()
def evaluate_sentiment(text: List[Dict], **kwargs) -> List:
    """
    Reference:
    - https://huggingface.co/docs/transformers/tasks/sequence_classification
    """

    # data preparation
    text = [obj['generated_text'] for obj in text]

    # sentiment analysis
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    logits = classifier(**inputs).logits

    prob = logits.softmax(dim=1).cpu()
    prob = prob[:, 1]
    # prob, label = prob.max(dim=1)

    # map to string label, then convert to Python List
    # label = list(map(lambda x: id2label.get(x), label.numpy().tolist()))
    prob = prob.numpy().tolist()

    # zip label and prob into a list of tuple
    # pred = list(zip(label, prob))

    # return (label, prob)
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
    # LLM generating strategy.
    beam_search = {
        'do_sample': False, 'num_beams': 5, 'num_return_sequences': 5, "no_repeat_ngram_size": 2,
    }

    # LLM generating strategy.
    nucleus_sampling = {
        'do_sample': True, 'top_p': 0.7, 'num_return_sequences': 1000,
    }

    def emit(candidates: List[Dict], prob: List[float]) -> List[Dict]:
        return [
            { 'generated_text': c['generated_text'], 'prob': f'{p:6f}' }
                for c, p in zip(candidates, prob)
        ]

    def savefig(benign: List, adversarial: List, title: str):
        plt.figure(figsize=(8, 8))

        vec = [benign[1], adversarial[1], ]
        plt.hist(vec, bins=20, alpha=0.5, label=['benign', 'adversarial'])

        plt.legend()
        plt.title(title)
        plt.savefig(f'{title}.png')
        plt.clf()

    # User defined prompt.
    prompt = "When you're thirsty and in need of a drink,"
    prepend = " to On by that"

    # Evaluate the baseline.
    prob = baseline(prompt)
    print(f'{prob=}')

    # Evaluate the beam-search strategy.
    benign, adversarial = evaluate_strategy(prompt, prepend, { 'max_length': 50, **beam_search })
    savefig(benign, adversarial, 'beam_search')

    # Store the result to json file.
    beam_search = {
        'param': beam_search, 'benign': emit(*benign), 'adversarial': emit(*adversarial),
    }

    # Evaluate the nucleus-sampling strategy.
    benign, adversarial = evaluate_strategy(prompt, prepend, { 'max_length': 50, **nucleus_sampling })
    savefig(benign, adversarial, 'nucleus_sampling')

    # Store the result to json file.
    nucleus_sampling = {
        'param': nucleus_sampling, 'benign': emit(*benign), 'adversarial': emit(*adversarial),
    }

    # Store the result to json file.
    with open('result.json', 'w') as f:
        json.dump({
            'prompt': prompt,
            'prepend': prepend,
            'strategy': {
                'beam_search': beam_search,
                'nucleus_sampling': nucleus_sampling,
            },
        }, f, indent=4, ensure_ascii=False)

    # save the figure to file
    # plt.legend()
    # plt.savefig('result.png')


if __name__ == '__main__':
    evaluate()
