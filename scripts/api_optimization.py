import argparse
import copy
import json
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pysnooper
import torch
import wandb

sys.path.append("../") # Support load module script from current directory
from scripts.image_optimization import RunTurbo, tuple_type
from utils.objectives.api_text_generation_objective import APITextGenerationObjective


class APIOptimizeText(RunTurbo):
    """
    The class inherits from OptimizeImage(RunTurbo), and slightly modifies
    the methods to fit the text generation task.
    """

    def __init__(self, args):
        self.args = args

    # @pysnooper.snoop()
    def get_baseline_prompts(self):
        """
        This function returns some prompts crafted manually.
        We expect that the attacker finds a better prompts to manually crafted prompts.
        """
        ## ** , here we maximize this "loss" !
        if self.args.loss_type not in ["log_prob_neg", "log_prob_pos"]:
            target_str = self.args.target_string # perc_target, num_target, "target_occurrences"
            prompts = [f"generate lots of {target_str}s", f"words with many {target_str}s"]
        else:
            prompts = [] # 5 example baseline prompts
            if self.args.loss_type == "log_prob_pos":
                target_str = "happy"
            elif self.args.loss_type == "log_prob_neg":
                target_str = "sad"
            else:
                assert 0

            # "happy happy happy happy"
            prompt1 = target_str
            for i in range(self.args.n_tokens - 1):
                prompt1 +=  f" {target_str}"
            prompts.append(prompt1)

            # "very very very happy"
            prompt2 = "very"
            for _ in range(self.args.n_tokens - 2):
                prompt2 += f" very"
            prompt2 +=  f" {target_str}"
            prompts.append(prompt2)

        # If prepend task:
        # if self.args.prepend_task:
        #     temp = []
        #     for prompt in prompts:
        #         temp.append(prompt + f" {self.args.prepend_to_text}")
        #     prompts = temp

        return prompts

    # @pysnooper.snoop()
    def log_baseline_prompts(self):
        # Get manually crafted prompts as baseline value.
        # Prompts optimized from TuRBO should be better than this one.
        baseline_prompts = self.get_baseline_prompts()

        # Clone the prompts until it fits the batch size
        while (len(baseline_prompts) % self.args.bsz) != 0:
            baseline_prompts.append(baseline_prompts[0])

        n_batches = int(len(baseline_prompts) / self.args.bsz)
        baseline_scores = []
        baseline_gen_text = []

        for i in range(n_batches):
            prompt_batch = baseline_prompts[i*self.args.bsz:(i+1)*self.args.bsz]
            out_dict = self.args.objective.pipe(
                input_type="prompt",
                input_value=prompt_batch,
                output_types=['generated_text', 'loss']
            )
            ys = out_dict['loss'].mean(-1)
            gen_text = out_dict["generated_text"]
            baseline_scores.append(ys)
            baseline_gen_text = baseline_gen_text + gen_text

        baseline_scores = torch.cat(baseline_scores).detach().cpu() # self.best_baseline_score
        self.best_baseline_score = baseline_scores.max().item()
        best_score_idx = torch.argmax(baseline_scores).item()

        # Rewrite this log function by tensorboard SummaryWriter
        # The extra 2 space in the string is for the markdown newline
        # See: https://stackoverflow.com/questions/45016458/tensorflow-tf-summary-text-and-linebreaks
        # self.tracker.writer.add_text('adversarial',
        #     f"{self.best_baseline_score:.4f}  \n" + '  \n'.join(baseline_gen_text[best_score_idx]), 0
        # )

        # Uncomment these lines to see what's log in origin code base.
        # self.tracker.log({
        #     "baseline_scores": baseline_scores.tolist(),
        #     "baseline_prompts": baseline_prompts,
        #     "baseline_gen_text": baseline_gen_text,
        #     "best_baseline_score": self.best_baseline_score,
        #     "best_baseline_prompt": baseline_prompts[best_score_idx],
        #     "best_baseline_gen_text": baseline_gen_text[best_score_idx],
        # })

        self.prcnt_latents_correct_class_most_probable = 1.0 # for compatibility with image gen task

    # @pysnooper.snoop()
    def get_init_data(self):
        # get scores for baseline_prompts
        self.log_baseline_prompts()

        # initialize random starting data
        YS = [] # scores
        XS = [] # embedding vectors
        PS = [] # prompts
        GS = [] # generated text

        # if do batches of more than 10, get OOM
        n_batches = math.ceil(self.args.n_init_pts / self.args.bsz)
        for ix in range(n_batches):
            X = torch.randn(self.args.bsz, self.args.objective.dim) * 0.01
            XS.append(X)

            prompts, ys, gen_text = self.args.objective(X.to(torch.float16))
            YS.append(ys)
            PS = PS + prompts
            GS = GS + gen_text

        Y = torch.cat(YS).detach().cpu()
        self.args.X = torch.cat(XS).float().detach().cpu()
        self.args.Y = Y.unsqueeze(-1)
        self.args.P = PS
        self.args.G = GS
        self.best_baseline_score = -1 # filler b/c used my image opt

    # @pysnooper.snoop()
    def save_stuff(self):
        # X = self.args.X
        Y = self.args.Y
        P = self.args.P
        G = self.args.G
        # best_x = X[Y.argmax(), :].squeeze().to(torch.float16)
        # torch.save(best_x, f"../best_xs/{wandb.run.name}-best-x.pt")

        best_prompt = P[Y.argmax()]
        # self.tracker.log({
        #     "best_prompt": best_prompt
        # })

        best_gen_text = G[Y.argmax()]
        # self.tracker.log({
        #     "best_gen_text": best_gen_text
        # })

        self.tracker.writer.add_text('best-prompt', best_prompt, self.args.objective.num_calls)
        self.tracker.writer.add_text('adversarial',
            '  \n'.join(best_gen_text), self.args.objective.num_calls
        )

    def call_oracle_and_update_next(self, x_next):
        p_next, y_next, g_next = self.args.objective(x_next.to(torch.float16))
        self.args.P = self.args.P + p_next # prompts
        self.args.G = self.args.G + g_next # generated text
        return y_next

    def init_objective(self,):
        self.args.objective = APITextGenerationObjective(
            num_gen_seq=self.args.num_gen_seq,
            max_gen_length=self.args.max_gen_length,
            dist_metric=self.args.dist_metric, # "sq_euclidean",
            n_tokens=self.args.n_tokens,
            minimize=self.args.minimize,
            batch_size=self.args.bsz,
            prepend_to_text=self.args.prepend_to_text,
            lb = self.args.lb,
            ub = self.args.ub,
            # text_gen_model=self.args.text_gen_model,
            loss_type=self.args.loss_type,
            # target_string=self.args.target_string,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_init_per_prompt', type=int, default=None )
    parser.add_argument('--n_init_pts', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.01 )
    parser.add_argument('--n_epochs', type=int, default=2)
    parser.add_argument('--init_n_epochs', type=int, default=80)
    parser.add_argument('--acq_func', type=str, default='ts' )
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--minimize', type=bool, default=False)
    parser.add_argument('--task', default="textgen")
    parser.add_argument('--hidden_dims', type=tuple_type, default="(256,128,64)")
    parser.add_argument('--more_hdims', type=bool, default=True) # for >8 tokens only
    parser.add_argument('--seed', type=int, default=1 )
    parser.add_argument('--success_value', type=int, default=8)
    parser.add_argument('--break_after_success', type=bool, default=False)

    # Increase to 10000 to see what will happen when attacker has more attack budgets,
    # default number is 3000.
    parser.add_argument('--max_n_calls', type=int, default=3000 )
    parser.add_argument('--num_gen_seq', type=int, default=5 )
    parser.add_argument('--max_gen_length', type=int, default=20 )
    parser.add_argument('--dist_metric', default="sq_euclidean" )
    parser.add_argument('--n_tokens', type=int, default=4 )
    parser.add_argument('--failure_tolerance', type=int, default=32 )
    parser.add_argument('--success_tolerance', type=int, default=10 )

    # Increased to 3000 to see what will happen without early-stop, default number is 1000.
    parser.add_argument('--max_allowed_calls_without_progress', type=int, default=1000 ) # for square baseline!
    parser.add_argument('--text_gen_model', default="api" )
    parser.add_argument('--square_attack', type=bool, default=False)
    parser.add_argument('--bsz', type=int, default=10)
    parser.add_argument('--prepend_task', type=bool, default=False)
    parser.add_argument('--prepend_to_text', default="I am happy")
    parser.add_argument('--loss_type', default="log_prob_pos" )
    parser.add_argument('--target_string', default="t" )
    parser.add_argument('--wandb_entity', default="nmaus" )
    parser.add_argument('--wandb_project_name', default="prompt-optimization-text" )
    args = parser.parse_args()

    if args.text_gen_model != "api":
        raise ValueError("Only support API text generation model.")


    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    abbrev = f"-{args.prepend_to_text.replace(' ', '')[:2]}" if args.prepend_to_text else ""
    root = '/tmp2/edwardlee/checkpoints/adversarial_prompting'
    root = os.path.join(root, f'{timestamp}{abbrev}')

    os.makedirs(root, exist_ok=True)

    with open(os.path.join(root, f"exp-{timestamp}.json"), "w") as f:
        for i in range(50):
            # random seed -> current time
            random.seed(None)

            # random a key from Mersenne Twister Pseudo-RNG
            seed = random.getrandbits(32)
            args.seed = seed

            runner = APIOptimizeText(args)
            runner.run()

            # Store the best prompts to another file
            best_score = runner.args.Y.max().item()
            best_prompt = runner.args.P[runner.args.Y.argmax()]

            # Save the best prompt to a json file
            json.dump({
                "best_score": best_score,
                "best_prompt": best_prompt,
                "prompts": runner.args.P,
                "scores": runner.args.Y.tolist(),
                "prepend_to_text": args.prepend_to_text,
                "seed": seed,
            }, f)

            f.flush()

            # save the vectors x and score y to a .pt file
            # torch.save({
            #     "x": runner.args.X,
            #     "y": runner.args.Y,
            # }, f"exp-{timestamp}-{i:02d}.pt")

