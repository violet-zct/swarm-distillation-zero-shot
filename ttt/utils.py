import numpy as np
import math


def compute_metrics(logprobs, num_examples, num_targets, num_prompts, golds=None, metrics=None):
    predictions = [[] for _ in range(num_prompts)]
    avg_ensemble_predictions = []
    idx = 0
    for eidx in range(num_examples):
        avg_probs = np.zeros(num_targets)
        for pidx in range(num_prompts):
            max_ll, pred_label = -np.inf, -1
            # actually, the number of labels of each prompt should be the same
            for ii in range(num_targets):
                if logprobs[idx] > max_ll:
                    max_ll, pred_label = logprobs[idx], ii
                avg_probs[ii] += math.exp(logprobs[ii])
                idx += 1
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
        all_preds = [pptm[k] for pptm in prompt_metrics]
        results["max_" + k] = round(np.max(all_preds), 3)
        results["median_" + k] = round(np.meadian(all_preds), 3)
        results["mean_" + k] = round(np.mean(all_preds), 3)
        results["min_" + k] = round(np.min(all_preds), 3)
        results["var_" + k] = round(np.var(all_preds), 3)

    for k, v in ensemble_metrics.items():
        results["ensemble_avg" + k] = v

    return results

def summarize_metrics(predictions, avg_ensemble_predictions, golds, metrics):
    prompt_metrics = []
    for ppred in predictions:
        prompt_metrics.append(metrics.compute(predictions=ppred, references=golds))
    ensemble_metrics = metrics.compute(predictions=avg_ensemble_predictions, references=golds)

    results = {}
    for k, v in prompt_metrics[0].items():
        all_preds = [pptm[k] for pptm in prompt_metrics]
        results["max_" + k] = round(np.max(all_preds), 3)
        results["median_" + k] = round(np.meadian(all_preds), 3)
        results["mean_" + k] = round(np.mean(all_preds), 3)
        results["min_" + k] = round(np.min(all_preds), 3)
        results["var_" + k] = round(np.var(all_preds), 3)

    for k, v in ensemble_metrics.items():
        results["ensemble_avg" + k] = v

    return results