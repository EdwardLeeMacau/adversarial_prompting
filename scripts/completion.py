import json
import os
from datetime import datetime
from time import sleep
from typing import List

import numpy as np
import openai
import torch
from torch.nn import Embedding
from tqdm import tqdm, trange

openai.organization = 'org-sDd4QQ3oY8PPgOtLXkBp9iO2'
openai.api_key = os.getenv('OPENAI_API_KEY')
# response = openai.Model.list()

# Reference:
#
# https://platform.openai.com/docs/models/gpt-3-5
# https://openai.com/pricing
# https://platform.openai.com/docs/api-reference/completions/create

engine = "text-ada-001"
# engine = "text-davinci-003"
prompt = "A joke by a Chinese stand-up comedian that loosely referenced a slogan used to describe the country's military,"

# Reference:
#
# https://openai.com/blog/new-and-improved-embedding-model
def query_embedding(words: List[str]):
    while True:
        try:
            # Call API to query embedding
            response = openai.Embedding.create(
                model="text-embedding-ada-002", input=words
            )

        except openai.error.PermissionError as e:
            print(e, engine)
            response = {}
            break

        except openai.error.RateLimitError as e:
            # wait for 60 seconds then try again
            print(e)
            sleep(60)
            continue

        break

    # write to file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f'../data/embedding/{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(response, f, ensure_ascii=False)

    return response

def complete(prompt=prompt):
    config = {
        "echo": True,           # Echo back the prompt in addition to the completion
        "n": 5,                 # Number of possible outputs to generate
        "max_tokens": 50,       # Maximum tokens per output
        "temperature": 0.0,     # Randomness of the output
    }

    # Call API to generate text
    completion = openai.Completion.create(
        engine=engine, prompt=prompt, **config
    )

    print(completion)

    # write to file
    completion['config'] = config
    completion['prompt'] = prompt
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f'../data/completion/{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(completion, f, ensure_ascii=False, indent=4)

@torch.no_grad()
def download_embedding():
    # Modify embedding needs disabling auto-grad
    #
    # Reference:
    # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    with open('../data/gpt2-vocab.json', 'r', encoding='utf-8') as f:
        vocab = list(json.load(f).keys())
        n = len(vocab)

    embedding = Embedding(n, 1536)
    pbar = tqdm(range(0, n, 1000), desc='Querying embedding')
    for i in pbar:
        subset = vocab[i:i+1000]
        response = query_embedding(subset)['data']

        z = np.array([obj['embedding'] for obj in response])
        z = torch.from_numpy(z)
        embedding.weight[i:i+1000] = z

    # save embedding
    torch.save(embedding, '../data/embedding.pt')

if __name__ == '__main__':
    # download_embedding()
    complete()
