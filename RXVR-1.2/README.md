# RXVR 1.2

### Overview

RXVR 1.2 is the next step for implementing Robossis xvr framework. This release evaluates patient-specific fine-tuned model across multiple synthetic X-ray views.

Key improvements include: Learning rate exploration to avoid local minima, ICP-based refinement applied at both initial and final poses, and training the model with bigger orientation and translational ranges.

### Purpose

Improve the model evaluation by adjusting the model parameters, adding the Optimizer to the synthetic X-ray.

### Scope

Train and fine-tune patient-specific model with different learning rates and bigger rotational/translational ranges.

Conduct multi-view evaluations (pure translation, translation + rotations).

Apply ICP refinement to initial and final pose estimates for an accuracy boost.

### Training model parameters

 "xvr", "train",
    "--inpath", input_path,
    "--outpath", output_path,
    "--r1", "-45", "45",
    "--r2", "-45", "45",
    "--r3", "-15", "15",
    "--tx", "-500", "500",
    "--ty", "-1000", "-400",
    "--tz", "-500", "500",
    "--sdd", "1000.00",
    "--height", "128",
    "--delx", "2.42",
    "--lr", "0.01",
    "--reverse_x_axis",             #
    "--orientation", "PA",
    "--batch_size", "100",
    "--n_epochs", "1000",
    "--n_batches_per_epoch", "100",
    "--project", "RXVR1.2",
    "--name", "RXVR1.2_training"

### Fine-tuning model parameters

"xvr", "finetune",
    "--inpath", input_path,
    "--outpath", output_path,
    "--ckptpath", ckpt_path,
    "--lr", "0.007",                            # Optional: learning rate (0.01,0.001,0.005,0.007)
    "--batch_size", "100",
    "--n_epochs", "25",
    "--n_batches_per_epoch", "50",
    "--rescale", "1.0",
    "--project", "RXVR1.2",                     # Optional: wandb project name
    "--name", "RXVR1.2_Finetuned_05"

### Results
**Proximal Femur**

Translation only: mTRE reduced from ~2 mm → 0.3–0.7 mm after ICP.

<img width="1040" height="433" alt="Screenshot 2025-09-03 160805" src="https://github.com/user-attachments/assets/050dea9c-a0a3-41a6-bf5f-1ea5c29c84a2" />

Complex rotations(+30° α,+45° β,+15° γ): reduced from ~2–6 mm → ≤1 mm after ICP.

Before ICP:

 <img width="998" height="418" alt="Screenshot 2025-09-05 103703" src="https://github.com/user-attachments/assets/1a99a159-2674-41f6-a3d2-79e72e6756d1" />

After ICP: 

<img width="993" height="423" alt="Screenshot 2025-09-05 103732" src="https://github.com/user-attachments/assets/9c4561b7-1ec8-4038-a2c2-2c5d41507836" />

**Distal Femur**

Translation only: reduced from ~2–3 mm → 0.3–1.0 mm after ICP.

<img width="1007" height="420" alt="Screenshot 2025-09-10 160251" src="https://github.com/user-attachments/assets/da75e6ce-0cfa-47af-acb6-e22db8e9bb71" />

Complex rotations(+30° α,+45° β,+15° γ): reduced from ~2–3 mm → 0.5–1.3 mm after ICP.

Before ICP:

<img width="1013" height="427" alt="Screenshot 2025-09-10 164627" src="https://github.com/user-attachments/assets/22a4326d-9da8-4ef6-a980-4e32a727ad88" />

After ICP:

<img width="1008" height="429" alt="Screenshot 2025-09-10 164649" src="https://github.com/user-attachments/assets/2e261dd8-fb21-4004-bae1-dd0abb29ece6" />

### Key Lessons

Proper alignment between the training model’s coordinate system and the input X-ray is essential for accurate results.

ICP integration is critical: consistently reduces mTRE by 50–90%.

Learning rate matters.

Fiducial marking variance can shift mTRE significantly.

### Next Steps

Apply to real C-arm X-rays with verified ground-truth poses.

Expand evaluation beyond synthetic DRRs.

Improve initialization strategies for faster and more reliable convergence.

Resample and process the real X-ray image.
