# Masked Dual-Editing Diffusion Models for Multi-Object Image Editing

## Abstract

Multi-object editing aims to modify multiple objects or regions in complex scenes while preserving structural coherence. This task faces significant challenges in scenarios involving overlapping or interacting objects: (1) Inaccurate localization of target objects due to attention misalignment, leading to incomplete or misplaced edits; (2) Attribute-object mismatch, where color or texture changes fail to align with intended regions due to cross-attention leakage, creating semantic conflicts (\textit{e.g.}, color bleeding into non-target areas). Existing methods struggle with these challenges: approaches relying on global cross-attention mechanisms suffer from attention dilution and spatial interference between objects, while mask-based methods fail to bind attributes to geometrically accurate regions due to feature entanglement in multi-object scenarios. To address these limitations, we propose a training-free, inference-stage optimization approach that enables precise localized image manipulation in complex multi-object scenes, named MDE-Edit. MDE-Edit optimizes the noise latent feature in diffusion models via two key losses: Object Alignment Loss (OAL) aligns multi-layer cross-attention with segmentation masks for precise object positioning, and Color Consistency Loss (CCL) amplifies target attribute attention within masks while suppressing leakage to adjacent regions. This dual-loss design ensures localized and coherent multi-object edits. Extensive experiments demonstrate that MDE-Edit outperforms state-of-the-art methods in editing accuracy and visual quality, offering a robust solution for complex multi-object image manipulation tasks.

## MDE-Edit Implementation

### Setup Environment
Our method is tested using cuda12.4 on a single A6000.
The preparation work mainly includes downloading the pre-trained model and configuring the environment.

```bash
conda create -n MDE python=3.8
conda activate MDE

pip install -r requirements.txt
```

We use Stable Diffusion v1-4 as backbone, please download from [Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v1-4) .

The `code/edit_multi.sh` provide the edit sample.

```bash
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
```
The result is saved at `code/result`.
