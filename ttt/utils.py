import numpy as np
import math
import os


def index_median(array):
    x = np.argsort(array)
    h = len(x) // 2
    # return (x[h] + x[h+1]) // 2 if len(x) % 2 == 0 else x[h]
    return h


def write_results_to_file(fout_name, all_prompt_metrics, all_prompt_predictions, avg_ensemble_metrics, avg_ensemble_preds, golds):
    for k, v in all_prompt_metrics[0].items():
        results = {}
        all_metrics = [pptm[k] * 100 for pptm in all_prompt_metrics]
        median_prompt = all_prompt_predictions[index_median(all_metrics)]
        max_prompt = all_prompt_predictions[np.argsort(all_metrics)[-1]]
        results["max_" + k] = round(np.max(all_metrics), 2)
        results["median_" + k] = round(np.median(all_metrics), 2)
        results["mean_" + k] = round(np.mean(all_metrics), 2)
        results["min_" + k] = round(np.min(all_metrics), 2)
        results["std_" + k] = round(np.std(all_metrics), 2)
        results["ensemble_" + k] = round(avg_ensemble_metrics[k]*100, 2)
        if fout_name.startswith("results"):
            fout_name = fout_name + ".{}".format(k)
        else:
            fout_name = os.path.join(fout_name, k)
        with open(fout_name, "w") as fout:
            fout.write(",".join(["{}={}".format(kk, vv) for kk, vv in results.items()]) + "\n")
            # output predictions of prompts for each example
            for ii in range(len(all_prompt_predictions[0])):
                s = ",".join(["gold={}".format(golds[ii]), "median={}".format(median_prompt[ii]), "max={}".format(max_prompt[ii]), "esemb={}".format(avg_ensemble_preds[ii])]) + ","
                s += " ".join([str(all_prompt_predictions[jj][ii]) for jj in range(len(all_prompt_predictions))])
                fout.write(s + "\n")


def compute_metrics(logprobs, num_examples, num_targets, num_prompts, golds=None, metrics=None, fout_name=None):
    predictions = [[] for _ in range(num_prompts)]
    avg_ensemble_predictions = []
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
            avg_probs += normalized_probs
            predictions[pidx].append(pred_label)
        avg_probs = avg_probs / num_prompts
        avg_ensemble_predictions.append(np.argmax(avg_probs))

    if num_examples == 1:
        return [ppt[0] for ppt in predictions], avg_ensemble_predictions[0]

    prompt_metrics = []
    for ppred in predictions:
        prompt_metrics.append(metrics.compute(predictions=ppred, references=golds))
    ensemble_metrics = metrics.compute(predictions=avg_ensemble_predictions, references=golds)

    results = {}
    for k, v in prompt_metrics[0].items():
        all_metrics = [pptm[k]*100 for pptm in prompt_metrics]
        results["max_" + k] = round(np.max(all_metrics), 2)
        results["median_" + k] = round(np.median(all_metrics), 2)
        results["mean_" + k] = round(np.mean(all_metrics), 2)
        results["min_" + k] = round(np.min(all_metrics), 2)
        results["std_" + k] = round(np.std(all_metrics), 2)

    for k, v in ensemble_metrics.items():
        results["ensemble_avg" + k] = round(v * 100, 2)

    write_results_to_file(fout_name, prompt_metrics, predictions, ensemble_metrics, avg_ensemble_predictions, golds)
    return results


def summarize_metrics(predictions, avg_ensemble_predictions, golds, metrics, fout_name=None):
    prompt_metrics = []
    for ppred in predictions:
        prompt_metrics.append(metrics.compute(predictions=ppred, references=golds))
    ensemble_metrics = metrics.compute(predictions=avg_ensemble_predictions, references=golds)

    results = {}
    for k, v in prompt_metrics[0].items():
        all_metrics = [pptm[k]*100 for pptm in prompt_metrics]
        results["max_" + k] = round(np.max(all_metrics), 2)
        results["median_" + k] = round(np.median(all_metrics), 2)
        results["mean_" + k] = round(np.mean(all_metrics), 2)
        results["min_" + k] = round(np.min(all_metrics), 2)
        results["std_" + k] = round(np.std(all_metrics), 2)

    for k, v in ensemble_metrics.items():
        results["ensemble_avg" + k] = round(v * 100, 2)

    if fout_name is not None:
        write_results_to_file(fout_name, prompt_metrics, predictions, ensemble_metrics, avg_ensemble_predictions, golds)
    return results