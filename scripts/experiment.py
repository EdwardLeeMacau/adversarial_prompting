"""
experiment.py

Learn how to use tiktokenizer
"""

import tiktoken
import json

ENCODING_NAME = 'text-ada-001'
ENCODING_FOR_MODEL = 'text-ada-001'

# encoding = tiktoken.get_encoding(ENCODING_NAME)
encoding = tiktoken.encoding_for_model(ENCODING_FOR_MODEL)

# print(dir(encoding))

# Dump this variable to json file
ranks = {encoding.decode([v]): v for _, v in encoding._mergeable_ranks.items()}

# for k in ranks.keys():
#     print(k)
#     # try:
#     #     print(k.decode('utf-8'))
#     # except:
#     #     print(k)
#     #     break

# print(ranks.keys())
# print(f'{ranks=}')

with open('mergeable_ranks.json', 'w', encoding='utf-8') as f:
    json.dump(encoding._mergeable_ranks, f, ensure_ascii=False, indent=4)

tokens = encoding.encode(' in at at at')
string = encoding.decode(tokens)
print(f'{tokens=}, {string=}')

tokens = encoding.encode('in at at at')
string = encoding.decode(tokens)
print(f'{tokens=}, {string=}')
