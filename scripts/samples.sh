#! /bin/bash

mkdir trials

python main.py -O --text "a simple plywood chair with planar surfaces" --workspace trials/chair --save_mesh --mcubes_resolution 128 --decimate_target 100 --num_steps 16 --max_steps 128 --albedo_iters 100 --density_thresh 10 --blob_density 0.2
python main.py -O --text "a simple plywood table with planar surfaces" --workspace trials/table --save_mesh --mcubes_resolution 128 --decimate_target 100 --num_steps 16 --max_steps 128 --albedo_iters 500 --density_thresh 1 --blob_density 1
python main.py -O --text "a simple plywood cabinet with planar surfaces" --workspace trials/cabinet --save_mesh
python main.py -O --text "a simple plywood shelve with planar surfaces" --workspace trials/shelve --save_mesh
python main.py -O --text "a simple plywood drawer with planar surfaces" --workspace trials/drawer --save_mesh
python main.py -O --text "a simple plywood desk with planar surfaces" --workspace trials/desk --save_mesh
python main.py -O --text "a simple plywood cradle with planar surfaces" --workspace trials/cradle --save_mesh
python main.py -O --text "a simple plywood bed with planar surfaces" --workspace trials/bed --save_mesh
python main.py -O --text "a simple plywood house with planar surfaces" --workspace trials/house --save_mesh --mcubes_resolution 128 --decimate_target 100 --num_steps 16 --max_steps 128 --albedo_iters 500 --density_thresh 1 --blob_density 1
