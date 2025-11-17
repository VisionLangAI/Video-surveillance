# evaluation/evaluate.py

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
import config

def evaluate(preds, labels):
    assert len(preds) == len(labels)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='macro')
    rec = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=config.CLASS_NAMES)

    print("Accuracy:", acc)
    print("Precision (macro):", prec)
    print("Recall (macro):", rec)
    print("F1 (macro):", f1)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=str, required=True, help="npz file containing preds and labels")
    args = parser.parse_args()
    data = np.load(args.preds)
    evaluate(data["preds"], data["labels"])
