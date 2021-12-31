import numpy as np
import scipy
import math
import os


def index_median(array):
    x = np.argsort(array)
    h = len(x) // 2
    # return (x[h] + x[h+1]) // 2 if len(x) % 2 == 0 else x[h]
    return h


def write_results_to_file(fout_name, suffix, all_prompt_metrics, all_prompt_predictions,
                          avg_ensemble_metrics, avg_ensemble_preds,
                          vote_ensemble_metrics, vote_ensemble_preds,
                          golds, avg_entropy=None):
    results = {}
    for k, v in all_prompt_metrics[0].items():
        all_metrics = [pptm[k] * 100 for pptm in all_prompt_metrics]
        median_prompt = all_prompt_predictions[index_median(all_metrics)]
        max_prompt = all_prompt_predictions[np.argsort(all_metrics)[-1]]
        results["max_" + k] = round(np.max(all_metrics), 2)
        results["median_" + k] = round(np.median(all_metrics), 2)
        results["mean_" + k] = round(np.mean(all_metrics), 2)
        results["min_" + k] = round(np.min(all_metrics), 2)
        results["std_" + k] = round(np.std(all_metrics), 2)
        results["avg_ensemble_" + k] = round(avg_ensemble_metrics[k]*100, 2)
        results["vote_ensemble_" + k] = round(vote_ensemble_metrics[k] * 100, 2)
        if fout_name.startswith("results"):
            nfout = fout_name + f".{k}_{suffix}"
        else:
            nfout = os.path.join(fout_name, f'{k}_{suffix}')
        with open(nfout, "w") as fout:
            fout.write(",".join(["{}={}".format(kk, vv) for kk, vv in results.items()]) + "\n")
            if avg_entropy is not None:
                fout.write("acc: " + " ".join([str(vv) for vv in all_metrics]) + "\n")
                fout.write("ent: " + " ".join([str(vv) for vv in avg_entropy]) + "\n")
            # output predictions of prompts for each example
            for ii in range(len(all_prompt_predictions[0])):
                s = ",".join(["gold={}".format(golds[ii]), "median={}".format(median_prompt[ii]), "max={}".format(max_prompt[ii]),
                              "avg_esemb={}".format(avg_ensemble_preds[ii]), "vote_esemb={}".format(vote_ensemble_preds[ii])]) + ","
                s += " ".join([str(all_prompt_predictions[jj][ii]) for jj in range(len(all_prompt_predictions))])
                fout.write(s + "\n")
    return results

def write_unsupervised_results_to_fille(fout_name, results, all_prompt_predictions, golds=None):
    with open(fout_name, "w") as fout:
        fout.write(",".join(["{}={}".format(kk, vv) for kk, vv in results.items()]) + "\n")

        # output predictions of prompts for each example
        for ii in range(len(all_prompt_predictions[0])):
            s = f"gold={golds[ii]}, " if golds is not None else ""
            s += " ".join([str(all_prompt_predictions[jj][ii]) for jj in range(len(all_prompt_predictions))])
            fout.write(s + "\n")


def compute_metrics(logprobs,
                    num_examples,
                    num_targets,
                    num_prompts,
                    golds=None,
                    metrics=None,
                    fout_name=None,
                    suffix=None,
                    pseudo_dist="smooth",
                    return_all_prompt_preds=False,
                    random_selection_ensemble=0.0,
                    **kwargs):
    predictions = [[] for _ in range(num_prompts)]
    entropies = [[] for _ in range(num_prompts)]
    avg_ensemble_predictions = []
    vote_ensemble_predictions = []
    all_avg_probs = []
    idx = 0
    for eidx in range(num_examples):
        avg_probs = np.zeros(num_targets)
        for pidx in range(num_prompts):
            max_ll, pred_label = -np.inf, -1
            # actually, the number of labels of each prompt should be the same
            normalized_probs = np.zeros(num_targets)
            for ii in range(num_targets):
                if logprobs[idx] > max_ll:
                    max_ll, pred_label = logprobs[idx], ii
                normalized_probs[ii] = math.exp(logprobs[idx])
                idx += 1
            normalized_probs = normalized_probs / normalized_probs.sum()
            entropies[pidx].append(-(normalized_probs * np.log(normalized_probs)).sum())
            avg_probs += normalized_probs
            all_avg_probs.append(normalized_probs)
            predictions[pidx].append(pred_label)

        # import pdb; pdb.set_trace()
        if 0.0 < random_selection_ensemble < 1.0 and num_examples == 1:
            selected_prompts = np.random.permutation(num_prompts)[:int(num_prompts * random_selection_ensemble)]
            avg_probs = sum([all_avg_probs[jj] for jj in selected_prompts]) / len(selected_prompts)
            all_preds = [predictions[jj][-1] for jj in selected_prompts]
        else:
            avg_probs = avg_probs / num_prompts
            all_preds = [ppt[-1] for ppt in predictions]

        avg_label = np.argmax(avg_probs)
        counts = [all_preds.count(ii) + 1 for ii in range(num_targets)]
        vote_label = np.argmax(counts)
        total = float(sum(counts))
        vote_probs = [c / total for c in counts]

        if return_all_prompt_preds and num_examples == 1:
            random_indices = np.random.permutation(len(all_avg_probs))
            avg_probs = [all_avg_probs[ii] for ii in random_indices]
            vote_probs = [[1 if c == predictions[ii][-1] else 0 for c in range(num_targets)] for ii in random_indices]
            return [ppt[0] for ppt in predictions], avg_probs, vote_probs

        if pseudo_dist == 'argmax':
            avg_probs = [1 if c == avg_label else 0 for c in range(num_targets)]
            vote_probs = [1 if c == vote_label else 0 for c in range(num_targets)]

        if num_examples == 1:
            avg_ensemble_predictions.append(avg_probs)
            vote_ensemble_predictions.append(vote_probs)
        else:
            avg_ensemble_predictions.append(avg_label)
            vote_ensemble_predictions.append(vote_label)

    if num_examples == 1:
        return [ppt[0] for ppt in predictions], avg_ensemble_predictions[0], vote_ensemble_predictions[0]

    prompt_metrics = []
    for ppred in predictions:
        prompt_metrics.append(metrics.compute(predictions=ppred, references=golds))
    avg_ensemble_metrics = metrics.compute(predictions=avg_ensemble_predictions, references=golds)
    avg_entropy = [np.mean(ents) for ents in entropies]
    vote_ensemble_metrics = metrics.compute(predictions=vote_ensemble_predictions, references=golds)

    results = write_results_to_file(fout_name, suffix, prompt_metrics, predictions,
                                    avg_ensemble_metrics, avg_ensemble_predictions,
                                    vote_ensemble_metrics, vote_ensemble_predictions, golds, avg_entropy)
    print(results)
    return results, None

def compute_entropy(predictions, num_targets):
    all_entropy = []
    for prompt_p in predictions:
        # import pdb; pdb.set_trace()
        prob = np.bincount(prompt_p, minlength=num_targets)
        prob = prob / len(prompt_p)
        all_entropy.append(scipy.stats.entropy(prob))

    return np.array(all_entropy)


def compute_unsupervised_metrics(logprobs,
                                 num_examples,
                                 num_targets,
                                 num_prompts,
                                 golds=None,
                                 metrics=None,
                                 fout_name=None,
                                 suffix=None,
                                 return_all_prompt_preds=False,
                                 random_selection_ensemble=0.0,
                                 initial_predictions=None,
                                 **kwargs):

    # import pdb; pdb.set_trace()
    predictions = [[] for _ in range(num_prompts)]
    entropies = [[] for _ in range(num_prompts)]
    avg_ensemble_predictions = []
    vote_ensemble_predictions = []
    all_avg_probs = []
    idx = 0
    for eidx in range(num_examples):
        avg_probs = np.zeros(num_targets)
        for pidx in range(num_prompts):
            max_ll, pred_label = -np.inf, -1
            # actually, the number of labels of each prompt should be the same
            normalized_probs = np.zeros(num_targets)
            for ii in range(num_targets):
                if logprobs[idx] > max_ll:
                    max_ll, pred_label = logprobs[idx], ii
                normalized_probs[ii] = math.exp(logprobs[idx])
                idx += 1
            normalized_probs = normalized_probs / normalized_probs.sum()
            entropies[pidx].append(-(normalized_probs * np.log(normalized_probs)).sum())
            avg_probs += normalized_probs
            all_avg_probs.append(normalized_probs)
            predictions[pidx].append(pred_label)

    results = {}

    entropy = compute_entropy(predictions, num_targets)
    results['all entropy'] = entropy
    results['avg entropy'] = entropy.mean()

    fout_name = os.path.join(fout_name, f'unsupervised_dev_{suffix}')

    if initial_predictions is None:
        print('finish collecting initial predictions before optimization')
        print(results)
        write_unsupervised_results_to_fille(fout_name, results, predictions, golds)
        return results, predictions
    else:
        initial_entropy = compute_entropy(initial_predictions, num_targets)
        results['delta all entropy'] = entropy - initial_entropy
        results['delta avg entropy'] = results['delta all entropy'].mean()
        print(results)
        write_unsupervised_results_to_fille(fout_name, results, predictions, golds)
        return results, None


def summarize_metrics(predictions, avg_ensemble_predictions, vote_ensemble_predictions, golds, metrics, fout_name=None):
    prompt_metrics = []
    for ppred in predictions:
        prompt_metrics.append(metrics.compute(predictions=ppred, references=golds))
    avg_ensemble_metrics = metrics.compute(predictions=avg_ensemble_predictions, references=golds)
    vote_ensemble_metrics = metrics.compute(predictions=vote_ensemble_predictions, references=golds)

    results = {}
    for k, v in prompt_metrics[0].items():
        all_metrics = [pptm[k]*100 for pptm in prompt_metrics]
        results["max_" + k] = round(np.max(all_metrics), 2)
        results["median_" + k] = round(np.median(all_metrics), 2)
        results["mean_" + k] = round(np.mean(all_metrics), 2)
        results["min_" + k] = round(np.min(all_metrics), 2)
        results["std_" + k] = round(np.std(all_metrics), 2)

    for k, v in avg_ensemble_metrics.items():
        results["avg_ensemble_avg" + k] = round(v * 100, 2)

    for k, v in vote_ensemble_metrics.items():
        results["vote_ensemble_avg" + k] = round(v * 100, 2)

    if fout_name is not None:
        _ = write_results_to_file(fout_name, "final", prompt_metrics, predictions,
                                  avg_ensemble_metrics, avg_ensemble_predictions,
                                  vote_ensemble_metrics, vote_ensemble_predictions, golds)
    return results


def compute_loss_scale(pred_labels, prompt_groups, group_id, answer_id):
    """
    compute how likely (unormalized) the prompts outside the current group supports the
    current answer
    """

    total = 0
    support = 0.
    # for prompt_id, pred in enumerate(pred_labels):
    #     if prompt_id not in prompt_groups[group_id]:
    #         total += 1
    #         if pred == answer_id:
    #             support += 1.
    for prompt_id, pred in enumerate(pred_labels):
        total += 1
        if pred == answer_id:
            support += 1.

    # only one group
    if total == 0:
        return 0

    return support

