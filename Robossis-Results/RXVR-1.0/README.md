# RXVR 1.0
### Overview

RXVR 1.0 is the first version of the Robossis XVR-based femur pose estimation framework. It leverages synthetic X-ray generation (DRRs) from the femur CT data (SE000002.nii) to train and evaluate models for C-arm source pose estimation.

Two strategies are explored:

1- **Fine-tuned patient-agnostic model**: Pretrained on pelvis CTs, using xvr model, then fine-tuned on femur CT (SE000002.nii).

2- **Patient-specific model**: trained from scratch on a single femur CT.

### Purpose

Benchmark patient-agnostic vs. patient-specific models for femur pose estimation.

Understand how pixel spacing, resolution, and training parameters affect accuracy.

### Scope

Synthetic dataset generation with DRR simulations from femur CT.

Model training & fine-tuning using Vivek’s XVR framework.

Evaluation across multiple resolutions (128×128 → 1436×1436).

Quantitative metrics: mean Target Registration Error (mTRE).


### Usage

1- **Fine-Tuning a Patient-Agnostic Model (Model B)**
xvr finetune --inpath SE000002.nii.gz --outpath output --epochs 10 --batch_size 116

Input: femur CT (SE000002.nii.gz)

Output: Finetuned_Patient_Agnostic.pth

2- **Training a Patient-Specific Model (Model A)**
xvr train --inpath SE000002.nii.gz --outpath output --epochs 230 --batch_size 32 --height 256 --delx 0.310547

Input: femur CT (SE000002.nii.gz)

Output: Model_trained_from_scratch.pth

### Results

1- **Fine-tuned patient-agnostic model (Model B):**

Best result: mTRE = 376.85 mm (256x256, 0.310547 mm spacing).
<img width="624" height="219" alt="Picture1" src="https://github.com/user-attachments/assets/cfa06ae6-ad61-422a-a250-99ada36e1223" />

2- **Patient-specific model (Model A):**

Best result: mTRE = 317.62 mm (256x256, 2.1764375 mm spacing).
<img width="624" height="216" alt="Picture2" src="https://github.com/user-attachments/assets/c527f5d9-70f5-4985-8cda-fc770b0430f3" />

### Next Step
Matching CT voxel spacing and DRR pixel spacing
