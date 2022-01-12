import numpy as np
import sys

model_fname = sys.argv[1]
if len(sys.argv) > 2:
    baseline_fname = sys.argv[2]
else :
    baseline_fname = None


def compute_consistency_test_set(fname):
    all_scores = []
    with open(fname) as fin:
        for line in fin:
            if line.startswith("gold"):
                fields = line.strip().split(",")
                ind_predictions = np.array([int(i) for i in fields[-1].split()])
                score, count = 0, 0
                for i in range(len(ind_predictions)):
                    for j in range(i, len(ind_predictions)):
                        count += 1
                        score += ind_predictions[i] == ind_predictions[j]
                score = score * 1.0 / count
                all_scores.append(score)
    return np.mean(all_scores)


print(compute_consistency_test_set(model_fname))

if baseline_fname is not None:
    pass
