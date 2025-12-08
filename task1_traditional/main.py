import os
import sys
import argparse
from dataset import load_data
from models import TraditionalClassifier
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser(description='Task 1: Traditional Classification')
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()

    X_train, y_train, selected_classes = load_data(args.data_root, split='train', num_classes=args.num_classes)
    X_val, y_val, _ = load_data(args.data_root, split='val', num_classes=args.num_classes, selected_classes=selected_classes)
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    model = TraditionalClassifier()
    model.train(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"\nResult:")
    print(f"Selected {len(selected_classes)} classes.")
    print(f"Validation Accuracy: {acc * 100:.2f}%")

if __name__ == '__main__':
    main()
