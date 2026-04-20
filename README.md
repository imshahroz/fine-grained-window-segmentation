# Fine-Grained Window Segmentation

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)
![Status](https://img.shields.io/badge/Project-Computer%20Vision-black)

A CNN-based computer vision project for **fine-grained car window segmentation** and **targeted background removal**.

## Project Poster
![Project Poster](assets/poster.png)

## Overview
Traditional background removal tools often treat the car as one solid object.  
This project focuses specifically on **window-region segmentation**, where reflections, glare, tint, and background scenery make the problem more challenging.

The goal is to isolate only the **transparent glass area** and clean the unwanted visible background through the windows.

## Key Idea
Instead of segmenting the whole vehicle, this pipeline predicts a **window mask** only.

## Pipeline
1. Load vehicle image
2. Resize and normalize input
3. Predict window mask using CNN
4. Refine mask with OpenCV post-processing
5. Export mask and cleaned output

## Tech Stack
- Python
- PyTorch
- OpenCV
- NumPy
- Pillow
- Matplotlib

## Folder Structure
```bash
fine-grained-window-segmentation/
│── data/
│   ├── images/
│   └── masks/
│── outputs/
│── assets/
│   └── poster.png
│── model.py
│── dataset.py
│── train.py
│── infer.py
│── utils.py
│── requirements.txt
│── .gitignore
│── README.md
