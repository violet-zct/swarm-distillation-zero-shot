import numpy as np
import sys

model_fname = sys.argv[1]
baseline_fname = sys.argv[2]
n_labels = int(sys.argv[3])


def fleiss_kappa(M):
    """Computes Fleiss' kappa for group of annotators.
    :param M: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number of subjects and 'k' = the number of categories.
        'M[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
    :type: numpy matrix
    :rtype: float
    :return: Fleiss' kappa score
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators
    tot_annotations = N * n_annotators  # the total # of annotations
    category_sum = np.sum(M, axis=0)  # the sum of each category over all items

    # chance agreement
    p = category_sum / tot_annotations  # the distribution of each category over all annotations
    PbarE = np.sum(p * p)  # average chance agreement over all categories

    # observed agreement
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N  # add all observed agreement chances per item and divide by amount of items

    return round((Pbar - PbarE) / (1 - PbarE), 4)


def pairwise_consistency(ind_predictions):
    score, count = 0, 0
    for i in range(len(ind_predictions)):
        for j in range(i, len(ind_predictions)):
            count += 1
            score += ind_predictions[i] == ind_predictions[j]
    score = score * 1.0 / count
    return score


def compute_consistency_test_set(fname):
    all_scores = []
    M = []
    with open(fname) as fin:
        for line in fin:
            if line.startswith("max_accuracy"):
                fields = line.strip().split(",")
                ensemble_res = ",".join(fields[-2:])

            if line.startswith("acc"):
                accuracies = [float(v) for v in line.strip().split()[1:]]

            if line.startswith("ent"):
                entropies = [float(v) for v in line.strip().split()[1:]]

            if line.startswith("gold"):
                fields = line.strip().split(",")
                ind_predictions = [int(i) for i in fields[-1].split()]

                counts = [ind_predictions.count(ii) for ii in range(n_labels)]
                M.append(counts)

                score = pairwise_consistency(ind_predictions)
                all_scores.append(score)
    pairwise_score = np.mean(all_scores)
    M = np.array(M)
    fleiss_kappa_score = fleiss_kappa(M)
    permutation = np.random.permutation(n_labels)
    check = fleiss_kappa(M[:, permutation])
    assert check == fleiss_kappa_score
    return ensemble_res, pairwise_score, fleiss_kappa_score, accuracies, entropies

baseline_res, baseline_pairwise, baseline_fk, baseline_accuracies, baseline_ents = compute_consistency_test_set(baseline_fname)
model_res, model_pairwise, model_fk, model_accuracies, model_ents = compute_consistency_test_set(model_fname)

print("baseline results")
print(baseline_res)
print("model results")
print(model_res)
print("fleiss karpa of baseline and model: {}, {}".format(baseline_fk, model_fk))
print("pairwise of baseline and model: {}, {}".format(baseline_pairwise, model_pairwise))

delta_accs = []
delta_ents = []
for bacc, bent, macc, ment in zip(baseline_accuracies, baseline_ents, model_accuracies, model_ents):
    delta_accs.append(macc-bacc)
    delta_ents.append(ment-bent)
print()
print("delta accs")
print(" ".join([str(d) for d in delta_accs]))
print("delta ents")
print(" ".join([str(d) for d in delta_ents]))