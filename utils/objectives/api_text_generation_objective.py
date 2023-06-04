import json
import os
import sys
from time import sleep
from datetime import datetime
from typing import Dict, List

import openai
import pysnooper
import torch
import tiktoken
from torch.nn.parameter import Parameter
from transformers import (DistilBertForSequenceClassification,
                          DistilBertTokenizer, GPT2Tokenizer, GPT2Model)

sys.path.append("../")
from utils.objective import Objective

engine = "text-ada-001"
encoding = tiktoken.encoding_for_model(engine)

openai.organization = 'org-ghlHp7yuXkkpPdB66B6gbmeJ'
openai.api_key = os.getenv('OPENAI_API_KEY')

class GPT3:
    def complete(self, prompt: List[str]):
        # Call API to generate text
        while True:
            try:
                completion = openai.Completion.create(
                    engine=engine, prompt=prompt, **self.config
                )

            except openai.error.APIError as e:
                print(e)
                sleep(60)
                continue

            except openai.error.RateLimitError as e:
                print(e)
                sleep(60)
                continue

            break

        # write to file
        completion['config'] = self.config
        completion['prompt'] = prompt

        # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # with open(f'../data/completion/{timestamp}.json', 'w', encoding='utf-8') as f:
        #     json.dump(completion, f, ensure_ascii=False, indent=4)

        return completion

    def __init__(self):
        self.max_length = 50
        self.num_return_sequences = 1
        self.temperature = 0.0
        self.echo = True

    @property
    def config(self):
        return {
            "echo": self.echo,                  # Echo back the prompt in addition to the completion
            "n": self.num_return_sequences,     # Number of possible outputs to generate
            "max_tokens": self.max_length,      # Maximum tokens per output
            "temperature": self.temperature,    # Randomness of the output
        }

    # define a fake _complete() function for debugging
    def _complete(self, prompt: List[str]):
        with open('../data/completion/20230530-160537.json', 'r', encoding='utf-8') as f:
            completion = json.load(f)

        return completion

    # @pysnooper.snoop()
    def forward(self, prompts: List[str], **kwargs):
        # Check the attachment for debugging
        #   - '20230530-155814.json'
        #   - '20230530-160537.json'
        completion = self.complete(prompts)['choices']
        N, stride = len(completion), self.num_return_sequences

        generated_texts = []
        for start in range(0, N, stride):
            end = start + stride
            generated_texts.append([
                {'generated_text': res['text']} for res in completion[start:end]
            ])

        return generated_texts

    def __call__(self, prompts: List[str], **kwargs):
        return self.forward(prompts, **kwargs)

class APITextGenerationObjective(Objective):
    @torch.no_grad()
    # @pysnooper.snoop()
    def __init__(
        self,
        num_calls=0,
        n_tokens=1,
        minimize=False,
        batch_size=10,
        prepend_to_text="",
        num_gen_seq=5,                  # replace n in API parameters
        max_gen_length=10,
        dist_metric="sq_euclidean",
        lb=None,
        ub=None,
        # text_gen_model="opt",
        loss_type="log_prob_neg", # log_prob_neg, log_prob_pos
        # target_string="t",
        **kwargs,
    ):
        super().__init__(
            num_calls=num_calls, task_id='adversarial4', dim=n_tokens*1536, lb=lb, ub=ub, **kwargs,
        )

        assert dist_metric in ['cosine_sim', "sq_euclidean"]
        assert not minimize

        # TODO: Check if we need tokenizer?
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.vocab = self.tokenizer.get_vocab()

        # self.target_string = None
        self.loss_type = loss_type
        self.prepend_to_text: str = prepend_to_text
        self.N_extra_prepend_tokens: str = len(self.prepend_to_text.split())
        self.dist_metric = dist_metric
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        # The following model is for sentiment analysis, it detects whether a
        # sentence is positive or negative.
        #
        # Dataset: https://huggingface.co/datasets/sst2
        # Model card: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
        self.distilBert_tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.distilBert_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        ).to(self.torch_device)

        # TODO: Replace self.generator by GPT-3 API.
        self.generator = GPT3()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.model = GPT2Model.from_pretrained('gpt2')
        self.word_embedder = self.model.get_input_embeddings()
        self.vocab = self.tokenizer.get_vocab()

        self.num_gen_seq = num_gen_seq
        self.max_gen_length = max_gen_length + n_tokens + self.N_extra_prepend_tokens

        self.all_token_idxs = list(self.vocab.values())

        # TODO: Load embedding layer from local file
        self.all_token_embeddings = self.word_embedder(torch.tensor(self.all_token_idxs)).to(self.torch_device)
        self.all_token_embeddings_norm = self.all_token_embeddings / self.all_token_embeddings.norm(dim=-1, keepdim=True)
        self.all_token_idxs = list(self.vocab.values())

        self.n_tokens = n_tokens
        # self.minmize = minimize
        self.batch_size = batch_size

        self.search_space_dim = 768
        self.dim = self.n_tokens * self.search_space_dim

    # @pysnooper.snoop()
    def proj_word_embedding(self, word_embedding: torch.Tensor) -> List[str]:
        '''
        Given a word embedding, project it to the closest word embedding of actual tokens using cosine similarity
        Iterates through each token dim and projects it to the closest token
        args:
            word_embedding: (batch_size, max_num_tokens, 1536) word embedding
        returns:
            proj_tokens: (batch_size, max_num_tokens) projected tokens
        '''
        assert self.dist_metric == "sq_euclidean"

        # Get word embeddings of all possible tokens as torch tensor
        proj_tokens = []

        # Iterate through batch_size
        for i in range(word_embedding.shape[0]):
            # Euclidean Norm
            dists = torch.norm(self.all_token_embeddings.unsqueeze(1) - word_embedding[i,:,:], dim = 2)

            closest_tokens = torch.argmin(dists, axis = 0)
            closest_tokens = torch.tensor(
                [self.all_token_idxs[token] for token in closest_tokens]
            ).cpu().numpy().tolist()
            closest_vocab = self.tokenizer.decode(closest_tokens)

            proj_tokens.append(closest_vocab)

        return proj_tokens

    # TODO: Modify generating strategy here.
    # @pysnooper.snoop()
    def prompt_to_text(self, prompts: List[str]) -> List[List[str]]:
        """ Given a list of prompts, generate texts using the generator. """
        attack_prompts = prompts

        if self.prepend_to_text:
            prompts = [cur_prompt + " " + self.prepend_to_text for cur_prompt in prompts]

        # query huggingface pipeline to generate texts
        gen_texts = self.generator(
            prompts, max_length=self.max_gen_length,
            num_return_sequences=self.num_gen_seq, num_beams=self.num_gen_seq,
            no_repeat_ngram_size=2, early_stopping=True,
        )

        # unwrap the generated texts
        gen_texts = [
            [cur_dict['generated_text'] for cur_dict in cur_gen] for cur_gen in gen_texts
        ]

        # hide the prepended text
        if self.prepend_to_text:
            gen_texts = [
                [text.split(atk + ' ')[1] for text in cur_gen] for cur_gen, atk in zip(gen_texts, attack_prompts)
            ]

        return gen_texts

    # @pysnooper.snoop()
    @torch.no_grad()
    def text_to_loss(self, text): # , loss_type='log_prob_pos')
        if self.loss_type not in ['log_prob_pos', 'log_prob_neg']:
            raise ValueError(f"loss_type must be one of ['log_prob_pos', 'log_prob_neg'] but was {self.loss_type}")

        # for attackers, they want to maximize the probability of the target sentiment.
        # therefore, when loss approaches to 0, the probability of the target sentiment is 1. (expected)
        # otherwise, when loss approaches to -inf, the probability of the target sentiment is 0.
        num_prompts = len(text)

        # TODO: Check what do these texts contain.
        flattened_text = [item for sublist in text for item in sublist]

        inputs = self.distilBert_tokenizer(flattened_text, return_tensors="pt", padding=True).to(self.torch_device)
        logits = self.distilBert_model(**inputs).logits
        probs = torch.softmax(logits, dim = 1)

        # Take target sentiment probability as objective score, attacker should
        # maximize it (if minimize=False)
        if self.loss_type == 'log_prob_pos':
            loss = torch.log(probs[:, 1])
        elif self.loss_type == 'log_prob_neg':
            loss = torch.log(probs[:, 0])
        else:
            raise ValueError(f"loss_type must be one of ['log_prob_pos', 'log_prob_neg'] but was {self.loss_type}")

        # keep_dim
        loss = loss.reshape(num_prompts, -1)

        return loss.cpu()  # torch.Size([2, 5]) = torch.Size([bsz, N_avg_over])

    # @pysnooper.snoop()
    def pipe(self, input_type, input_value, output_types):
        valid_input_types = ['raw_word_embedding' ,'prompt']
        valid_output_types = ['prompt', 'generated_text', 'loss']

        # Check that types are valid
        if input_type not in valid_input_types:
            raise ValueError(f"input_type must be one of {valid_input_types} but was {input_type}")

        for cur_output_type in output_types:
            if cur_output_type not in valid_output_types:
                raise ValueError(f"output_type must be one of {valid_output_types} but was {cur_output_type}")

        # Check that output is downstream
        pipeline_order = [
            "raw_word_embedding", "prompt", "generated_text", "loss"
        ]
        pipeline_maps = {
            "raw_word_embedding": self.proj_word_embedding,
            "prompt": self.prompt_to_text, # prompt to generated text
            "generated_text": self.text_to_loss, # text to generated loss
        }

        start_index = pipeline_order.index(input_type)
        max_end_index = start_index
        for cur_output_type in output_types:
            cur_end_index = pipeline_order.index(cur_output_type)
            if start_index >= cur_end_index:
                raise ValueError(f"{cur_output_type} is not downstream of {input_type}.")
            else:
                max_end_index = max(max_end_index,cur_end_index)

        # Pipeline is the composition of functions.
        # embedding -> proj_word_embedding -> prompt_to_text -> text_to_loss
        cur_pipe_val = input_value
        output_dict = {}
        for i in range(start_index, max_end_index):
            cur_type = pipeline_order[i]
            mapping = pipeline_maps[cur_type]
            cur_pipe_val = mapping(cur_pipe_val)
            next_type = pipeline_order[i+1]
            if next_type in output_types:
                output_dict[next_type] = cur_pipe_val

        return output_dict

    def query_oracle(self, x):
        """ Derived implementation of __call__ method from base class. """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float16)

        x = x.cuda()
        x = x.reshape(-1, self.n_tokens, self.search_space_dim)
        out_dict = self.pipe(
            input_type="raw_word_embedding",
            input_value=x,
            output_types=['prompt','generated_text','loss']
        )

        # Take average loss over generated texts
        y = out_dict['loss']
        y = y.mean(-1)

        return out_dict['prompt'], y, out_dict["generated_text"]

