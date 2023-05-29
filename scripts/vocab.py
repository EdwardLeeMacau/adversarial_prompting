"""
vocab.py

A script to dump the vocabulary of the GPT2 tokenizer to json file
"""

import json
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
vocab = tokenizer.get_vocab()

with open('gpt2-vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=4)
