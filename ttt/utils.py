from email.policy import default
import numpy as np
import scipy
import math
import os
import json
from collections import defaultdict


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


def write_unsupervised_results_to_file(fout_name, results, all_prompt_predictions, golds=None):
    with open(fout_name, "w") as fout:
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                fout.write("{}={}".format(key, " ".join([str(kk) for kk in value])) + "\n")
            else:
                fout.write("{}={}".format(key, value) + "\n")

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
                    self_train=False,
                    **kwargs):
    predictions = [[] for _ in range(num_prompts)]
    entropies = [[] for _ in range(num_prompts)]
    avg_ensemble_predictions = []
    vote_ensemble_predictions = []
    all_avg_probs = []  # only used when num of examples=1
    idx = 0
    logits = [[] for _ in range(num_prompts)]
    for eidx in range(num_examples):
        avg_probs = np.zeros(num_targets)
        for pidx in range(num_prompts):
            max_ll, pred_label = -np.inf, -1
            # actually, the number of labels of each prompt should be the same
            normalized_probs = np.zeros(num_targets)
            logit = []
            for ii in range(num_targets):
                if logprobs[idx] > max_ll:
                    max_ll, pred_label = logprobs[idx], ii
                normalized_probs[ii] = math.exp(logprobs[idx])
                logit.append(logprobs[idx])
                idx += 1
            logits[pidx].append(logit)
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
            if not self_train:
                random_indices = np.random.permutation(len(all_avg_probs))
            else:
                random_indices = np.arange(len(all_avg_probs))
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

    # print logits
    if fout_name.startswith("results"):
        nfout = fout_name + ".logits.p"
    else:
        nfout = os.path.join(fout_name, f'logits.{suffix}.p')
    for pidx in range(num_prompts):
        with open("{}{}".format(nfout, pidx), "w") as fout:
            for logit in logits[pidx]:
                fout.write(" ".join([str(l) for l in logit]) + "\n")

    results = write_results_to_file(fout_name, suffix, prompt_metrics, predictions,
                                    avg_ensemble_metrics, avg_ensemble_predictions,
                                    vote_ensemble_metrics, vote_ensemble_predictions, golds, avg_entropy)
    print(results)
    return results, None


def compute_metrics_simple(logprobs,
                           num_examples,
                           num_targets,
                           num_prompts,
                           golds=None,
                           metrics=None,
                           fout_name=None,
                           suffix=None,
                           pseudo_dist="smooth",
                           **kwargs):
    predictions = [[] for _ in range(num_prompts)]
    entropies = [[] for _ in range(num_prompts)]
    pred_probs = [[] for _ in range(num_examples)]
    avg_ensemble_predictions = []
    vote_ensemble_predictions = []
    all_avg_probs = []  # only used when num of examples=1
    idx = 0
    logits = [[] for _ in range(num_prompts)]
    for eidx in range(num_examples):
        avg_probs = np.zeros(num_targets)
        for pidx in range(num_prompts):
            max_ll, pred_label = -np.inf, -1
            # actually, the number of labels of each prompt should be the same
            normalized_probs = np.zeros(num_targets)
            logit = []
            for ii in range(num_targets):
                if logprobs[idx] > max_ll:
                    max_ll, pred_label = logprobs[idx], ii
                normalized_probs[ii] = math.exp(logprobs[idx])
                logit.append(logprobs[idx])
                idx += 1
            logits[pidx].append(logit)
            normalized_probs = normalized_probs / normalized_probs.sum()
            pred_probs[eidx].append(normalized_probs)
            entropies[pidx].append(-(normalized_probs * np.log(normalized_probs)).sum())
            avg_probs += normalized_probs
            all_avg_probs.append(normalized_probs)
            predictions[pidx].append(pred_label)

        # import pdb; pdb.set_trace()
        avg_probs = avg_probs / num_prompts
        all_preds = [ppt[-1] for ppt in predictions]

        avg_label = np.argmax(avg_probs)
        counts = [all_preds.count(ii) + 1 for ii in range(num_targets)]
        vote_label = np.argmax(counts)
        total = float(sum(counts))
        vote_probs = [c / total for c in counts]

        if pseudo_dist == 'argmax':
            avg_probs = [1 if c == avg_label else 0 for c in range(num_targets)]
            vote_probs = [1 if c == vote_label else 0 for c in range(num_targets)]

        avg_ensemble_predictions.append(avg_label)
        vote_ensemble_predictions.append(vote_label)

    prompt_metrics = []
    for ppred in predictions:
        prompt_metrics.append(metrics.compute(predictions=ppred, references=golds))
    avg_ensemble_metrics = metrics.compute(predictions=avg_ensemble_predictions, references=golds)
    avg_entropy = [np.mean(ents) for ents in entropies]
    vote_ensemble_metrics = metrics.compute(predictions=vote_ensemble_predictions, references=golds)

    results = write_results_to_file_v2(fout_name, prompt_metrics, predictions,
                                    avg_ensemble_metrics, avg_ensemble_predictions,
                                    vote_ensemble_metrics, vote_ensemble_predictions, golds, avg_entropy,
                                    pred_probs)
    print(results)
    return results, None



def write_results_to_file_v2(fout_name, all_prompt_metrics, all_prompt_predictions,
                            avg_ensemble_metrics, avg_ensemble_preds,
                            vote_ensemble_metrics, vote_ensemble_preds,
                            golds, avg_entropy=None, pred_probs=None):
    results = {}
    num_prompts, num_examples = len(pred_probs[0]), len(pred_probs)
    fout = open(fout_name, "w")
    fout.write("num_prompts={},num_examples={}\n".format(num_prompts, num_examples))
    
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

        fout.write("{}: {} {} {} {} {} {} {}\n".format(fout_name, num_prompts, results["max_" + k], results["median_" + k],
                                                  results["mean_" + k], results["min_" + k], results["std_" + k], results["avg_ensemble_" + k]))
        fout.write(",".join(["{}={}".format(kk, vv) for kk, vv in results.items()]) + "\n")
        fout.write("acc:\t" + " ".join([str(vv) for vv in all_metrics]) + "\n")
        fout.write("ent:\t" + " ".join([str(vv) for vv in avg_entropy]) + "\n")

        print("number of examples = {}".format(num_examples))

        fout.write("==================\n")
        # output predictions of prompts for each example
        for ii in range(len(all_prompt_predictions[0])):
            s = ",".join(["eid={}".format(ii), "gold={}".format(golds[ii]), "median={}".format(median_prompt[ii]), "max={}".format(max_prompt[ii]),
                          "avg_esemb={}".format(avg_ensemble_preds[ii]), "vote_esemb={}".format(vote_ensemble_preds[ii])]) + ","
            s += " ".join([str(all_prompt_predictions[jj][ii]) for jj in range(len(all_prompt_predictions))])
            fout.write(s + "\n")
            for jj in range(len(pred_probs[0])):
                fout.write(" ".join([str(round(pp, 4)) for pp in pred_probs[ii][jj]]) + "\n")
            fout.write("========================\n")
        fout.close()
    return results

def ece_score(py, y_test, n_bins=10):
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)

def compute_metrics_train(logprobs,
                          num_examples,
                          prompt_info,
                          fout_name,):

    data = []
    classification_prompts_probs = defaultdict(list)
    classification_prompts_golds = defaultdict(list)
    prev_idx = -1
    this_example = []
    for sidx, edix, idx, gidx, label, pname, pin, pout, choices in prompt_info:
        if idx != prev_idx:
            prev_idx = idx
            if len(this_example) > 0:
                data.append(this_example)
            this_example = []
        this_logprob = logprobs[sidx:edix]
        if choices is not None and len(choices) > 1:
            normalized_probs = np.exp(this_logprob)
            normalized_probs = normalized_probs / normalized_probs.sum()
            classification_prompts_probs[pname].append(normalized_probs)
            classification_prompts_golds[pname].append(label)
        this_example.append({"prompt_name": pname, "pinput": pin, "poutput": pout, "group_idx": gidx, "label": label, "choices": choices, 
                             "seq_log_probs": [float(ll) for ll in this_logprob]})
    
    data.append(this_example)
    classification_prompt_scores = {}
    all_accs = []
    max_acc, min_acc, median_acc = 0, 1, 0
    for pname in classification_prompts_golds.keys():
        probs = np.array(classification_prompts_probs[pname])
        preds = np.argmax(probs, axis=1)
        golds = classification_prompts_golds[pname]
        acc = np.equal(preds, golds).astype(np.float32).mean()
        ece = ece_score(probs, golds, n_bins=20)
        classification_prompt_scores[pname] = {"acc": float(acc), "ece": float(ece)}
        max_acc = max(max_acc, acc)
        min_acc = min(min_acc, acc)
        all_accs.append(acc)
    median_acc = all_accs[index_median(all_accs)]
    print("max_acc = {}, min_acc = {}, median_acc = {}".format(max_acc, min_acc, median_acc))
    
    print(len(data))
    print(data[0])
    for example in data:
        for pp in example:
            pp["acc"] = classification_prompt_scores[pp["prompt_name"]]["acc"] if pp["prompt_name"] in classification_prompt_scores else None
            pp["ece"] = classification_prompt_scores[pp["prompt_name"]]["ece"] if pp["prompt_name"] in classification_prompt_scores else None
    print(data[0]) 
    with open("{}.json".format(fout_name), "w") as fout:
        json.dump(data, fout)
    return classification_prompt_scores

def print_dict(dd):
    for key, value in dd.items():
        if isinstance(value, np.ndarray):
            print("{}: {}".format(key, " ".join([str(kk) for kk in value])))
        else:
            print("{}: {}".format(key, value))


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
    all_avg_probs = [[] for _ in range(num_prompts)]
    idx = 0
    for eidx in range(num_examples):
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
            all_avg_probs[pidx].append(normalized_probs)
            predictions[pidx].append(pred_label)

    results = {}

    entropy = compute_entropy(predictions, num_targets)
    results['all entropy'] = entropy
    results['avg entropy'] = entropy.mean()
    all_continuous_entropy = []
    for probs in all_avg_probs:
        all_continuous_entropy.append(scipy.stats.entropy(np.mean(probs, 0)))
    results['avg cont entropy'] = np.mean(all_continuous_entropy)

    fout_name = os.path.join(fout_name, f'unsupervised_dev_{suffix}')

    if initial_predictions is None:
        print('finish collecting initial predictions before optimization')
        print_dict(results)
        write_unsupervised_results_to_file(fout_name, results, predictions, golds)
        return results, predictions
    else:
        initial_entropy = compute_entropy(initial_predictions, num_targets)
        results['delta all entropy'] = entropy - initial_entropy
        results['delta avg entropy'] = results['delta all entropy'].mean()
        print_dict(results)
        write_unsupervised_results_to_file(fout_name, results, predictions, golds)
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


def compute_unsupervised_dev_best_results(dir_path, min_train_steps, metrics=['avg entropy', 'avg cont entropy']):
    unsup_dev_prefix = "unsupervised_dev_"
    eval_prefix = "accuracy_"
    all_checkpoints = []
    for fname in os.listdir(dir_path):
        if eval_prefix in fname:
            all_checkpoints.append(int(fname.split("_")[-1]))
    all_checkpoints.sort()

    best_ens_acc = 0.0
    best_ckpt = 0
    best_dev_results = {}
    all_results = {}
    for ckpt in all_checkpoints:
        with open(os.path.join(dir_path, eval_prefix+str(ckpt))) as fin:
            line = fin.readline()
            all_results[ckpt] = line.strip()
            line = line.strip().split(",")
            for field in line:
                k, v = field.split("=")
                v = float(v)
                if k == "avg_ensemble_accuracy" or k == "vote_ensemble_accuracy":
                    if v > best_ens_acc:
                        best_ckpt = ckpt
                        best_ens_acc = v

        if ckpt <= min_train_steps:
            continue
        if not os.path.exists(os.path.join(dir_path, unsup_dev_prefix+str(ckpt))):
            continue

        with open(os.path.join(dir_path, unsup_dev_prefix+str(ckpt))) as fin:
            for line in fin:
                if line.startswith("gold"):
                    break
                for metric in metrics:
                    if line.startswith(metric):
                        value = float(line.strip().split("=")[-1])
                        if metric in best_dev_results:
                            # larger metric is better: entropy
                            if value > best_dev_results[metric][-1]:
                                best_dev_results[metric] = (ckpt, value)
                        else:
                            best_dev_results[metric] = (ckpt, value)
    print("Best checkpoint at step {}: ".format(best_ckpt))
    print(all_results[best_ckpt])
    for k, v in best_dev_results.items():
        print("Best checkpoint selected by {} at step {}:".format(k, v[0]))
        print(all_results[v[0]])
