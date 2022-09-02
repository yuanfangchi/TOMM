#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/tep/"
vocab_dir="datasets/data_preprocessed/tep/vocab"
total_iterations=2000
path_length=3
hidden_size=2
embedding_size=2
batch_size=42
beta=0.1
Lambda=0.02
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
base_output_dir="output/tep/"
model_load_dir="nothing"
load_model=0
nell_evaluation=0
distributed_training=1
split_random=0
transferred_training=1