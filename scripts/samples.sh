#! /bin/bash

mkdir trials

CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a DSLR photo of a plywood chair" --workspace trials/chair
CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a DSLR photo of a plywood table" --workspace trials/table
CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a DSLR photo of a wooden house" --workspace trials/house