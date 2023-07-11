from typing import Dict, Optional, Sequence, Set

import numpy as np


def get_accuracy(outputs) -> float:
    num_correct = 0
    num_total = 0
    for output in outputs:
        num_total += 1
        num_correct += int(output["match"])
    if num_total == 0:
        return float("nan")
    else:
        return num_correct / num_total


def get_confusion_matrix(outputs: Sequence[Dict], class_labels: Optional[Set] = None):
    labels = set()
    for r in outputs:
        labels.add(r["expected"])
    if class_labels is None:
        labels = {label: i for i, label in enumerate(sorted(labels))}
    else:
        assert labels.issubset(class_labels)
        labels = {label: i for i, label in enumerate(class_labels)}
    result = np.zeros((len(labels), len(labels) + 1), dtype=int)
    for r in outputs:
        i = labels[r["expected"]]
        j = labels.get(r["picked"], len(labels))
        result[i, j] += 1
    return result


def compute_precision(confusion_matrix, idx=0):
    return confusion_matrix[idx, idx] / confusion_matrix[:, idx].sum()


def compute_recall(confusion_matrix, idx=0):
    return confusion_matrix[idx, idx] / confusion_matrix[idx, :].sum()


def compute_f_score(confusion_matrix, idx=0, beta=1.0):
    precision = compute_precision(confusion_matrix, idx=idx)
    recall = compute_recall(confusion_matrix, idx=idx)
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def compute_averaged_f_score(confusion_matrix, beta=1.0, average="macro"):
    assert average in ["macro"]
    f_scores = []
    for i in range(confusion_matrix.shape[0]):
        f_scores.append(compute_f_score(confusion_matrix, idx=i, beta=beta))
    return np.array(f_scores).mean()
