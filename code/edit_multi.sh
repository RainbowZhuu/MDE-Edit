#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 edit.py --source_prompt="a cat wearing a straw hat" \
                --target_prompt="a dog wearing a red hat" \
                --img_path="examples/1/cat_hat.png" \
                --oal_mask "examples/1/cat.png" \
                --idx2words="1:dog" \
                --ccl_mask "examples/1/hat.png" \
                --ccl_word "red" \
                --result_dir="result" \
                --max_iteration=15 \
                --scale=1.5 \
                --optimization_step=20