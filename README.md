# Histopathology Tumor Classification – MVA Challenge

This repository contains the code and experiments for our solution to the **Kaggle DATA Challenge**, conducted as part of the course *Deep Learning for Medical Image Analysis* of the **MVA Master's program**.

## Challenge Overview

The goal is to classify histopathological image patches as either **containing a tumor** (`label 1`) or **tumor-free** (`label 0`). A core difficulty is the **domain shift** between image acquisition centers. The dataset includes images from **5 medical centers**:
- **3 for training** (~100,000 images),
- **1 for validation** (~35,000 images),
- **1 for testing** (~85,000 images, labels hidden).

This domain variability makes it essential to address center-induced shifts in color and texture.

##  Summary of Our Approach

We combine **foundation models** for image embedding with both **learning-based** and **non-learning-based** domain adaptation strategies. Our pipeline involves:
- Feature extraction using **DINOv2** and **MedImageInsight**,
- Preprocessing via techniques like **grayscale**, **Macenko stain normalization**, or **CycleGAN**,
- Classification using a custom MLP trained on the embeddings.

##  Project Structure

- `src/`: core code for preprocessing, training, and submitting predictions.
- `CycleGAN/`: implementation of the MultiStain-CycleGAN model (adapted from external repo).
- `Notebooks/`: Colab notebooks to run the training, embedding extraction, and submission on GPU.
- `requirements.txt`: Python package dependencies.
- `.gitignore`: ignored files and folders for version control.
- `report.pdf`: final technical report explaining the methodology, experiments, and results.

## Acknowledgements

The `CycleGAN/` code is based on [DBO-DKFZ/multistain_cyclegan_normalization](https://github.com/DBO-DKFZ/multistain_cyclegan_normalization), with significant modifications to fit our framework.

---

For more details, please refer to the report and notebooks in the `Notebooks/` folder.
