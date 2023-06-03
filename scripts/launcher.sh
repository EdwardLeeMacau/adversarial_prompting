#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python text_optimization.py \
    --loss_type log_prob_pos \
    --prepend_task True \
    --max_gen_length 50 \
    --prepend_to_text "A joke by a Chinese stand-up comedian that loosely referenced a slogan used to describe the country's military" &

sleep 10;

CUDA_VISIBLE_DEVICES=0 python text_optimization.py \
    --loss_type log_prob_pos \
    --prepend_task True \
    --max_gen_length 50 \
    --prepend_to_text "Those within esports often get tired with constant comparisons with traditional sports," &

sleep 10;

CUDA_VISIBLE_DEVICES=1 python text_optimization.py \
    --loss_type log_prob_pos \
    --prepend_task True \
    --max_gen_length 50 \
    --prepend_to_text "You're exhausted, your body yawning for sleep." &

sleep 10;

CUDA_VISIBLE_DEVICES=1 python text_optimization.py \
    --loss_type log_prob_pos \
    --prepend_task True \
    --max_gen_length 50 \
    --prepend_to_text "Many people dread the blaring sound of an alarm clock," &

sleep 10;

CUDA_VISIBLE_DEVICES=2 python text_optimization.py \
    --loss_type log_prob_pos \
    --prepend_task True \
    --max_gen_length 50 \
    --prepend_to_text "We leave behind our fears of monsters under the bed as we say goodbye to our childhoods," &

sleep 10;

CUDA_VISIBLE_DEVICES=2 python text_optimization.py \
    --loss_type log_prob_pos \
    --prepend_task True \
    --max_gen_length 50 \
    --prepend_to_text "When you're thirsty and in need of a drink," &

sleep 10;

CUDA_VISIBLE_DEVICES=3 python text_optimization.py \
    --loss_type log_prob_pos \
    --prepend_task True \
    --max_gen_length 50 \
    --prepend_to_text "Don't use sugar substitutes if you are trying to lose weight," &

sleep 10;

CUDA_VISIBLE_DEVICES=3 python text_optimization.py \
    --loss_type log_prob_pos \
    --prepend_task True \
    --max_gen_length 50 \
    --prepend_to_text "Teens on social media is a problem many parents and guardians have lost sleep over," &

sleep 10;
