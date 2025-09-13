# RISE-MICCAI LISA 2025 Challenge Submission
## Vicent Caselles Ballester

This repository contains my submission to the RISE-MICCAI version of the 2025 Low-field pediatric brain magnetic resonance Image Segmentation and quality Assurance (LISA) Challenge, which focused on tasks 2a and 2b from the main LISA 2025 Challenge. Both tasks required the segmentation of brain structures from low-field MRI (from 0.064T Hyperfine SWOOP scanners).

More precisely, task 2a entailed the segmentation of the bilateral hippocampi, while task 2b focused on the following basal ganglia structures: the *caudate nucleus* and the *nucleus lentiformis* (*putamen* + *pallidum*).

### Proposed methodology

The methodology I used in my submission to the challenge had three main steps:
- Brain segmentation using HD-BET to reduce image dimensions (Isensee *et al.*).
- Denoising using a *classic* total variation denoising algorithm (Chambolle *et al.*).
- Synthetically generating a T1-w using `SynthSR` (Iglesias *et al.*).
- Feeding the two-channel input (lowfield + synthetic scan) to a simple UNet.

I trained a single model to perform both tasks, since I believe this helps the model have more spatially consistent outputs. Also, for the same reasons, I included the ventricles as a target during training since they were provided with the training data (although they are not the objective of the challenge).

The convolutional neural network was configured using a reverse-engineered breakdown of nnUNet's code (Isensee et al.), i.e. using patch-based training. With a 12 GB NVIDIA GeForce RTX 2080 Ti I had available as a target, this yielded the following configuration:
* Patch size: (128, 160, 128)
* Resolution stages: 6 (bottleneck shape: (4, 5, 4)).
* Oversampling on 50% of batches.
* 1000 epochs training.

To see a full breakdown of the training configuration see the [`debug.json`](trained_models/LISA_trained_models_28_08_25/debug.json) file. I also trained a fully-residual UNet (w encoder and decoder both with residual connections), which I couldn't submit. However, I am providing the resulting models for transparency -- also consult the corresponding json file if you're curious about the full description of the model.

### Reproducibility

To reproduce the full experiment follow the next steps:

### References
**Automated brain extraction of multi-sequence MRI using artificial neural
networks** \
Isensee F, Schell M, Tursunova I, Brugnara G, Bonekamp D, Neuberger U, Wick A,
Schlemmer HP, Heiland S, Wick W, Bendszus M, Maier-Hein KH, Kickingereder P. \ 
Hum Brain Mapp. 2019; 1–13. [[link](https://doi.org/10.1002/hbm.24750)]

**An algorithm for total variation minimization and applications** \
Chambolle A \
Journal of Mathematical Imaging and Vision, Springer, 2004, 20, 89-97.

**Joint super-resolution and synthesis of 1 mm isotropic MP-RAGE volumes from clinical MRI exams with scans of different 
orientation, resolution and contrast** \
Iglesias JE, Billot B, Balbastre Y, Tabari A, Conklin J, RG Gonzalez, Alexander DC, Golland P, Edlow B, Fischl B \
NeuroImage (2021) [[link](https://www.sciencedirect.com/science/article/pii/S1053811921004833)]
