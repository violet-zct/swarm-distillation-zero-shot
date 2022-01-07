import sys
import itertools
import numpy as np
import scipy

fprefix = sys.argv[1]
nprompts = int(sys.argv[2])

logits = [[] for _ in range(nprompts)]

for pidx in range(nprompts):
    with open(fprefix+".p{}.logits") as fin:
        for line in fin:
            logits[pidx].append([float(x) for x in line.strip().split()])


nlabels = len(logits[0][0])
nexamples = len(logits[0])
gold_labels = []

with open(fprefix+".accuracy_None") as fin:
    for line in fin:
        if line.startswith("gold"):
            label = int(line.strip().split(",")[0].split("=")[-1])
            gold_labels.append(label)


bstart, bend, interval = 0.1, 10, 0.2
biases = [list(range(bstart, bend, interval)) for _ in range(nlabels)]
biases = itertools.product(*biases)

entropies = []
avg_prob_ens_accuracies = []
vote_ens_accuracies = []

for bias in biases:
    avg_prob_ensemble_preds = []
    vote_ensemble_preds = []
    prompt_predictions = [[] for _ in range(nprompts)]
    for ii in range(nexamples):
        avg_probs = np.zeros(nlabels)
        preds = []
        for pidx in range(nprompts):
            logit = np.array(logits[pidx][ii]) + np.array(bias)
            normalized_probs = np.exp(logit)
            normalized_probs = normalized_probs / normalized_probs.sum()
            avg_probs += normalized_probs
            preds.append(np.argmax(logit))
            prompt_predictions[pidx].append(np.argmax(logit))
        avg_probs = avg_probs / nprompts
        avg_prob_ensemble_preds.append(np.argmax(avg_probs))
        counts = [preds.count(ii) + 1 for ii in range(nlabels)]
        vote_ensemble_preds.append(np.argmax(counts))
    soft_acc = sum(np.equal(avg_prob_ensemble_preds, gold_labels)) * 1.0 / nexamples
    hard_acc = sum(np.equal(vote_ensemble_preds, gold_labels)) * 1.0 / nexamples
    avg_prob_ens_accuracies.append(soft_acc)
    vote_ens_accuracies.append(hard_acc)
    all_entropy = []
    for prompt_p in prompt_predictions:
        # import pdb; pdb.set_trace()
        prob = np.bincount(prompt_p, minlength=nlabels)
        prob = prob / len(prompt_p)
        all_entropy.append(scipy.stats.entropy(prob))
    entropies.append(np.mean(all_entropy))

idx = np.argmax(entropies)
avg_prob_ens_result = avg_prob_ens_accuracies[idx]
vote_ens_result = vote_ens_accuracies[idx]

print("max entropy = {}, avg prob ensemble accuracy = {}, vote accuracy = {}".format(entropies[idx],
                                                                                     avg_prob_ens_result,
                                                                                     vote_ens_result))

