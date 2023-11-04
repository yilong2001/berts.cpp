#!/bin/bash

model=$1

python3 bert-classify-to-ggml ${model} 0
python3 bert-classify-to-ggml ${model} 1
