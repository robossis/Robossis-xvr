# RXVR 1.1

### Overview
RXVR 1.1 extends the Robossis XVR framework by focusing on a consistent field of view (FOV) and evaluating model performance not only on synthetic DRRs but also on real C-arm X-ray.

### Purpose
Improve upon RXVR 1.0 by stabilizing training with recommended parameters.

Maintain anatomical consistency across experiments by fixing FOV.

Evaluate patient-specific and fine-tuned models on both synthetic X-rays and real C-arm images.

### Scope

Training patient-specific models with constant FOV scaling.

Fine-tuning pretrained models under the same conditions.

Evaluation on:

  Synthetic X-rays at multiple resolutions (128–1024).

  Real C-arm femur X-ray (980×980 px, 0.316 mm spacing).  

### Results

**Synthetic X-Rays (constant FOV = 309.76 mm)**

**Patient-specific model:**

At 1024×1024, mTRE = 602.79 mm (final), 549.61 mm (initial).
<img width="1252" height="487" alt="Final_Pose" src="https://github.com/user-attachments/assets/34086265-5ce9-4ff2-bcba-65fc82cc156f" />
<img width="1251" height="492" alt="Initial_Pose" src="https://github.com/user-attachments/assets/52ea41ea-6419-4165-98a7-7ecf6bce126c" />

**Fine-tuned model:**

At 1024×1024, mTRE = 612.20 mm (final), 579.40 mm (initial)
<img width="1261" height="492" alt="Initial_Pose" src="https://github.com/user-attachments/assets/7a1cef09-5768-4bfa-9f1a-a89606078923" />
<img width="1260" height="488" alt="Final_Pose" src="https://github.com/user-attachments/assets/70e1b1ac-e593-4a93-adde-103c96fd193a" />

**Real C-Arm X-Ray (980×980 px, delx = 0.316)**
Predictions showed large positional errors.

Model localized femur shaft instead of proximal femur.

Registration attempts using register.py failed due to negative Y-offset and coordinate misalignment.
<img width="1258" height="455" alt="Final_Pose" src="https://github.com/user-attachments/assets/34e32677-8979-451d-a157-f4c864465a27" />

### Key Lessons
Keeping FOV constant improves anatomical scaling consistency.

Despite improved training stability, models degrade on real C-arm X-rays due to scatter, noise, and lack of intrinsic metadata.

### Next Steps
Revisit coordinate system alignment (volume, world, camera).

Improve initial pose estimates for registration.

Expand training coverage with wider rotation/translation ranges.

Optimize hyperparameters (learning rate, epochs) for robustness
