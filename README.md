
# PRML Project: Image Classification

This repository contains the coursework for "Pattern Recognition and Machine Learning," focusing on bird image classification using the CUB-200 dataset. The project consists of two main tasks:

- **Task 1: Traditional Method** — Implements a hand-written multiclass SVM (One-vs-Rest) using pre-extracted features to classify 10 bird species.
- **Task 2: Deep Learning Method** — Trains a ResNet-18 network from scratch with data augmentation for end-to-end classification of 200 bird species.

The project provides complete data processing, model training, evaluation, and a LaTeX report for easy reproduction and performance comparison between the two approaches.

## Project Structure

```
.
├── data/                   # Place the dataset here (train/ and val/ folders)
├── task1_traditional/      # Task 1: Traditional Pattern Recognition
│   ├── main.py             # Entry point for Task 1
│   ├── dataset.py          # Data loading for Task 1
│   └── models.py           # Traditional model implementations
├── task2_deep_learning/    # Task 2: Deep Neural Network
│   ├── main.py             # Entry point for Task 2
│   ├── dataset.py          # Custom Dataset class
│   ├── model.py            # CNN Model architecture
│   └── trainer.py          # Training and evaluation logic
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data:**

    Download the dataset from Tsinghua Cloud:

    - [data.zip (Tsinghua Cloud)](https://cloud.tsinghua.edu.cn/f/72aab178f61948c095dd/)

    After downloading, extract `data.zip` into the project root directory:

    ```bash
    unzip data.zip -d ./
    ```

    The final structure should look like:
    ```
    data/
    ├── train/
    │   ├── 001.Black_footed_Albatross/
    │   │   ├── ...jpg
    │   │   └── ...pt
    │   └── ...
    └── val/
        ├── 001.Black_footed_Albatross/
        │   ├── ...jpg
        │   └── ...pt
        └── ...
    ```

## Task 1: Traditional Method

Run the traditional classification task (10-class classification using feature vectors):

```bash
cd task1_traditional
python main.py
```

## Task 2: Deep Learning Method

Run the deep learning classification task (200-class classification using raw images):

```bash
cd task2_deep_learning
python main.py
```
