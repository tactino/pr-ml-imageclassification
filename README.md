# PRML Project: Image Classification

This project contains the implementation for the Pattern Recognition and Machine Learning course assignment.

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
    Download the dataset and extract it into the `data/` directory. The structure should look like:
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
