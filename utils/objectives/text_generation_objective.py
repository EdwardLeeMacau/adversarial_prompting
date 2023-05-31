import sys

import pysnooper
import torch
from typing import List
from transformers import (DistilBertForSequenceClassification,
                          DistilBertTokenizer, GPT2Tokenizer, OPTModel, GPT2Model,
                          pipeline)

sys.path.append("../")
from utils.objective import Objective

# TODO: Create derive class `APIObjective`
class TextGenerationObjective(Objective):
    @torch.no_grad()
    def __init__(
        self,
        num_calls=0,
        n_tokens=1,
        minimize=False,
        batch_size=10,
        prepend_to_text="",
        num_gen_seq=5,
        max_gen_length=10,
        dist_metric="sq_euclidean",
        lb=None,
        ub=None,
        text_gen_model="opt",
        loss_type="log_prob_neg", # log_prob_neg, log_prob_pos
        target_string="t",
        **kwargs,
    ):
        super().__init__(
            num_calls=num_calls, task_id='adversarial4', dim=n_tokens*768, lb=lb, ub=ub, **kwargs,
        )

        assert dist_metric in ['cosine_sim', "sq_euclidean"]
        assert not minimize

        # TODO: Extend here to attack other models
        # find models here: https://huggingface.co/models?sort=downloads&search=facebook%2Fopt
        if text_gen_model == "opt":
            model_string = "facebook/opt-125m"
        elif text_gen_model == "opt13b":
            model_string = "facebook/opt-13b"
        elif text_gen_model == "opt66b":
            model_string = "facebook/opt-66b"
        elif text_gen_model == "opt350":
            model_string = "facebook/opt-350m"
        elif text_gen_model == "gpt2":
            model_string = "gpt2"
        else:
            assert 0


        self.target_string = target_string
        self.loss_type = loss_type
        self.prepend_to_text = prepend_to_text
        self.N_extra_prepend_tokens = len(self.prepend_to_text.split())
        self.dist_metric = dist_metric
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_string)

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

        # Remove "Setting pad_token_id to eos_token_id:50256 for open-end generation." #12020
        # Reference: https://github.com/huggingface/transformers/issues/12020#issuecomment-898899723
        #
        # Leads the following warning:
        # /home/edwardleemacau/.pyenv/versions/3.9.16/lib/python3.9/site-packages/transformers/generation/utils.py:1219:
        # UserWarning: You have modified the pretrained model configuration to control generation.
        # This is a deprecated strategy to control generation and will be removed soon, in a future version.
        # Please use a generation configuration file
        # (see https://huggingface.co/docs/transformers/main_classes/text_generation)
        #
        # <transformers.pipelines.text_generation.TextGenerationPipeline object at 0x7f92b342dfa0>
        self.generator = pipeline("text-generation", model=model_string, device=0)
        self.generator.model.config.pad_token_id = self.generator.model.config.eos_token_id

        # This IS expected when running the script:
        #
        # "Some weights of the model checkpoint at facebook/opt-125m were not used when
        # initializing OPTModel: ['lm_head.weight']"
        #
        # Because we only need the models embeddings.
        if "opt" in model_string:
            self.model = OPTModel.from_pretrained(model_string)
        else:
            self.model = GPT2Model.from_pretrained(model_string)

        self.model = self.model.to(self.torch_device)

        # get embedded vectors and supported vocab here.
        # we do not know the exact word embedding and vocab set when attacking GPT-4,
        # thus we need to "guess" it. (consider transferability)
        self.word_embedder = self.model.get_input_embeddings()
        self.vocab = self.tokenizer.get_vocab()

        self.num_gen_seq = num_gen_seq
        self.max_gen_length = max_gen_length + n_tokens + self.N_extra_prepend_tokens

        if self.loss_type not in ['log_prob_pos', 'log_prob_neg']:
            self.related_vocab = [self.target_string]
            self.all_token_idxs = self.get_non_related_values()
        else:
            self.all_token_idxs = list(self.vocab.values())

        self.all_token_embeddings = self.word_embedder(torch.tensor(self.all_token_idxs).to(self.torch_device))
        self.all_token_embeddings_norm = self.all_token_embeddings / self.all_token_embeddings.norm(dim=-1, keepdim=True)
        self.n_tokens = n_tokens
        self.minmize = minimize
        self.batch_size = batch_size

        self.search_space_dim = 768
        self.dim = self.n_tokens * self.search_space_dim

    def get_non_related_values(self):
        """
        Consider prompting the LM to generate sentences with specific words or letters.
        This function collect the words supported by LM, that do not have the target string in it.
        """
        tmp = []
        for word in self.related_vocab:
            tmp.append(word)
            tmp.append(word+'</w>')
        self.related_vocab = tmp
        non_related_values = []
        for key in self.vocab.keys():
            if not key in self.related_vocab:
                non_related_values.append(self.vocab[key])
        return non_related_values

    # @pysnooper.snoop()
    def proj_word_embedding(self, word_embedding) -> List[str]:
        '''
        Given a word embedding, project it to the closest word embedding of actual tokens using cosine similarity
        Iterates through each token dim and projects it to the closest token
        args:
            word_embedding: (batch_size, max_num_tokens, 768) word embedding
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
            ).to(self.torch_device)
            closest_vocab = self.tokenizer.decode(closest_tokens)

            # cur_proj_tokens = [closest_vocab]
            proj_tokens.append(closest_vocab)  # cur_proj_tokens)

        return proj_tokens

    # TODO: Modify generating strategy here.
    def prompt_to_text(self, prompts: List[str]) -> List[List[str]]:
        """ Given a list of prompts, generate texts using the generator. """
        attack_prompts = prompts

        if self.prepend_to_text:
            prompts = [cur_prompt + " " + self.prepend_to_text for cur_prompt in prompts]

        # LLM generating strategy.
        beam_search = {
            'do_sample': False, 'num_beams': self.num_gen_seq,
            'num_return_sequences': self.num_gen_seq, "no_repeat_ngram_size": 2,
        }

        # LLM generating strategy.
        nucleus_sampling = {
            'do_sample': True, 'top_p': 0.7, 'num_return_sequences': self.num_gen_seq,
        }

        # query huggingface pipeline to generate texts
        gen_texts = self.generator(
            prompts, max_length=self.max_gen_length, early_stopping=True, **beam_search
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
        if self.loss_type in ['log_prob_pos', 'log_prob_neg']:
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
                loss = torch.log(probs[:,1])
            elif self.loss_type == 'log_prob_neg':
                loss = torch.log(probs[:,0])
            else:
                raise ValueError(f"loss_type must be one of ['log_prob_pos', 'log_prob_neg'] but was {self.loss_type}")

            # keep_dim
            loss = loss.reshape(num_prompts, -1)

        elif self.loss_type in ["perc_target", "num_target", "target_occurrences"]: # else: #s if self.loss_type == 'perc_ts':
            n_input = self.n_tokens + self.N_extra_prepend_tokens
            losses = []
            for outputs in text:
                scores_for_prompt = []
                for output in outputs:
                    words_with_target = 0.0
                    total_words = 0.0
                    occurrences = 0.0
                    for word in output.split()[n_input:]:
                        if self.target_string in word:
                            words_with_target += 1.0
                            for char in word:
                                if char == self.target_string:
                                    occurrences += 1.0
                        total_words += 1.0
                    if self.loss_type == "perc_target":
                        if total_words > 0:
                            score = words_with_target/total_words
                        else:
                            score = 0.0
                    elif self.loss_type == "num_target": # num words_with_target
                        score = words_with_target
                    elif self.loss_type == "target_occurrences": # total number of chars
                        score = occurrences
                    scores_for_prompt.append(score)
                scores_for_prompt = torch.tensor(scores_for_prompt).float()
                losses.append(scores_for_prompt.unsqueeze(0))
            loss = torch.cat(losses)
        else:
            assert 0

        return loss.cpu()  # torch.Size([2, 5]) = torch.Size([bsz, N_avg_over])

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
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float16)

        x = x.cuda()
        x = x.reshape(-1, self.n_tokens, self.search_space_dim)
        out_dict = self.pipe(
            input_type="raw_word_embedding",
            input_value=x,
            output_types=['prompt', 'generated_text', 'loss']
        )

        # Take average loss over generated texts
        y = out_dict['loss'].mean(-1)

        # negate the loss if we are minimizing the objective score
        if self.minmize:
            y = -y

        return out_dict['prompt'], y, out_dict["generated_text"]

