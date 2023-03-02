#!/usr/bin/env bash

# VisDA-2017
python3 test.py --cfg configs/visda2017.yaml --source synthetic --target real --weight_path log/visda2017/synthetic2real/final_model_synthetic2real.pth GPU_ID 6
python3 test.py --cfg configs/home.yaml --source RealWorld --target Art --weight_path log/home/RealWorld2Art/final_model_RealWorld2Art.pth GPU_ID 6
