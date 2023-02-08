#! /bin/bash

mkdir trials

CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a simple plywood chair with planar surfaces" --workspace trials/chair --save_mesh --mcubes_resolution 128 --decimate_target 100 --num_steps 16 --max_steps 128 --albedo_iters 500 --density_thresh 1 --blob_density 1
CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a simple plywood table with planar surfaces" --workspace trials/table --save_mesh --mcubes_resolution 128 --decimate_target 100 --num_steps 16 --max_steps 128 --albedo_iters 500 --density_thresh 1 --blob_density 1
CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a simple plywood house with planar surfaces" --workspace trials/house --save_mesh --mcubes_resolution 128 --decimate_target 100 --num_steps 16 --max_steps 128 --albedo_iters 500 --density_thresh 1 --blob_density 1