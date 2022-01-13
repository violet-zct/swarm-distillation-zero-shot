import os

import numpy as np
import sys
import scipy.stats

dirname = sys.argv[1]
n_labels = int(sys.argv[2])


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


def compute_entropy(predictions, num_targets):
    all_entropy = []
    for prompt_p in predictions:
        prob = np.bincount(prompt_p, minlength=num_targets)
        prob = prob / len(prompt_p)
        all_entropy.append(scipy.stats.entropy(prob))

    return np.array(all_entropy)


def compute_consistency_test_set(fname):
    all_scores = []
    M = []
    with open(fname) as fin:
        for line in fin:
            if line.startswith("max_accuracy"):
                fields = line.strip().split(",")
                ensemble_res = ",".join(fields[-2:])

            if line.startswith("gold"):
                fields = line.strip().split(",")
                ind_predictions = [int(i) for i in fields[-1].split()]

                counts = [ind_predictions.count(ii) + 1 for ii in range(n_labels)]
                M.append(counts)

                score = pairwise_consistency(ind_predictions)
                all_scores.append(score)
    pairwise_score = np.mean(all_scores)
    M = np.array(M)
    fleiss_kappa_score = fleiss_kappa(M)
    permutation = np.random.permutation(n_labels)
    check = fleiss_kappa_score(M[:, permutation])
    assert check == fleiss_kappa_score
    return ensemble_res, pairwise_score, fleiss_kappa_score


def read_unsupervised_metric(fname, metrics):
    with open(fname) as fin:
        for line in fin:
            if line.startswith("gold"):
                break
            for metric in ["avg entropy", "avg cont entropy"]:
                if line.startswith(metric):
                    value = float(line.strip().split("=")[-1])
                    if metric not in metrics:
                        metrics[metric] = [value]
                    else:
                        metrics[metric].append(value)
    return metrics


unsup_dev_prefix = "unsupervised_dev_"
eval_prefix = "accuracy_"
all_checkpoints = []
for fname in os.listdir(dirname):
    if eval_prefix in fname:
        all_checkpoints.append(int(fname.split("_")[-1]))
all_checkpoints.sort()

metrics = {}
ensemble_results, pairwise_consist_scores, fleiss_kappa_scores = [], [], []
for ii, cidx in enumerate(all_checkpoints):
    ensemble_res, pairwise_score, fleiss_kappa_score = compute_consistency_test_set(os.path.join(dirname, eval_prefix+str(cidx)))
    ensemble_results.append(ensemble_res)
    pairwise_consist_scores.append(pairwise_score)
    fleiss_kappa_scores.append(fleiss_kappa_score)

    read_unsupervised_metric(os.path.join(dirname, unsup_dev_prefix+str(cidx)), metrics)

    # mnames = list(metrics.keys())
    # s = "ckpt {}: {}, pairwise score={}, fleiss karpa={}, {}={}, {}={}".format(cidx, ensemble_res, pairwise_score,
    #                                                                            fleiss_kappa_score, mnames[0],
    #                                                                            metrics[mnames[0]][-1], mnames[1],
    #                                                                            metrics[mnames[1]][-1]
    #                                                                            )
    # print(s)


for i in range(len(all_checkpoints)-1):
    ensemble_res = ensemble_results[i]
    pairwise_score = pairwise_consist_scores[i]
    fleiss_kappa_score = fleiss_kappa_scores[i]

    delta_pairwise = pairwise_score - ensemble_results[i+1]
    delta_fleiss_karpa = fleiss_kappa_score - fleiss_kappa_scores[i+1]

    avg_ent = metrics["avg entropy"][i]
    avg_cont_ent = metrics["avg cont entropy"][i]

    delta_avg_ent = avg_ent - metrics["avg entropy"][i+1]
    delta_avg_cont_ent = avg_cont_ent - metrics["avg cont entropy"][i+1]

    s = "ckpt {}: {}, pairwise={}, delta pairwise={}, fleiss karpa={}, delta fk={}, " \
        "avg entropy={}, delta avg ent={}, avg cont entropy={}, delta cont ent={}".format(
        i, ensemble_res, pairwise_score, delta_pairwise, fleiss_kappa_score, delta_fleiss_karpa, avg_ent, delta_avg_ent,
        avg_cont_ent, delta_avg_cont_ent)
    print(s)