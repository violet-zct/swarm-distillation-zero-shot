from unittest.mock import NonCallableMock
from promptsource.templates import DatasetTemplates
import datasets
from torch.utils.data import Dataset
import numpy as np
import os
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class AdhocDatasetByPrompt(Dataset):
    # used for test, maybe need to extend this class for open-ended generation tasks
    def __init__(self, dataset_name, subset_name, testset_name, cache_dir, tokenizer, split=None, testdev_set=True):
        super().__init__()
        self.cache_dir = cache_dir

        self.split = split
        self.testdev_set = testdev_set  # if True, always use all prompts for ensemble predictions
        self.DATASET_NAME = dataset_name
        self.SUBSET_NAME = subset_name if subset_name != "none" else None
        self.TESTSET_NAME = testset_name
        self.PROMPTSET_NAME = dataset_name # has subset name?
        self.cb_surgery = False
        self.load()

        self.tokenizer = tokenizer

        self.prompts = DatasetTemplates(self.PROMPTSET_NAME, self.SUBSET_NAME)
        self.original_task_prompts = self.extract_original_task_prompts(check_valid_prompts=self.SUBSET_NAME=="copa")

        self.construct_meta_info()
        self.num_choices = len(self.prompts[self.original_task_prompts[0]].get_answer_choices_list(self.dataset[0]))
        print("{} has {} original task prompts, number choices = {}, total test examples = {}".format(self.DATASET_NAME + ("/" + self.SUBSET_NAME) if self.SUBSET_NAME is not None else "",
                                                                                 len(self.original_task_prompts),
                                                                                 self.num_choices,
                                                                                 len(self)))

    def load(self):
        if self.DATASET_NAME == "story_cloze":
            self.dataset = datasets.load_dataset("csv", data_files=os.path.join(self.cache_dir, "cloze_2016_val.csv"))["train"]
        else:
            self.dataset = datasets.load_dataset(self.DATASET_NAME, self.SUBSET_NAME,
                                             cache_dir=self.cache_dir)[self.TESTSET_NAME if self.split is None else self.split]

    @property
    def num_prompts(self):
        return len(self.original_task_prompts)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        inputs = []
        outputs = []
        prev_label = None
        for pname in self.original_task_prompts:
            input_template, output_template = self.prompts[pname].apply(item)
            # is output_template always the answer_choices[label]
            targets = self.prompts[pname].get_answer_choices_list(item)
            label = targets.index(output_template.strip())
            assert prev_label is None or label == prev_label
            prev_label = label
            for answer in targets:
                inputs.append(input_template.strip())
                outputs.append(answer.strip())
            self.set_num_choices(len(targets))

        # return inputs, outputs, item['label']
        if self.SUBSET_NAME == "cb" and self.cb_surgery:
            model_inputs = self.tokenizer(inputs, padding=False, truncation=True)
        else:
            model_inputs = self.tokenizer(inputs, padding=False, truncation=True, add_special_tokens=False)
        outputs_ids = self.tokenizer(outputs,  padding=False, truncation=True).input_ids
        model_inputs['labels'] = [[l if l != self.tokenizer.pad_token_id else -100 for l in x] for x in outputs_ids]

        results = [{
            'input_ids': input_id,
            'attention_mask': amask,
            'labels': label,
        } for input_id, amask, label in zip(model_inputs['input_ids'], model_inputs['attention_mask'], model_inputs['labels'])]
        return results, label

    def construct_meta_info(self):
        answer_groups = defaultdict(list)

        for pidx, pname in enumerate(self.original_task_prompts):
            answer_choices = self.prompts[pname].get_fixed_answer_choices_list()
            answer_choices = "_".join(answer_choices) if answer_choices is not None else None
            answer_groups[answer_choices].append(pidx)
        self.prompt_groups = [answer_groups[key] for key in answer_groups.keys()]

    def set_num_choices(self, n):
        check = True if self.num_choices == -1 else self.num_choices == n
        assert check
        self.num_choices = n

    def extract_original_task_prompts(self, check_valid_prompts=False):
        all_prompt_names = self.prompts.all_template_names
        invalid_names = []
        if check_valid_prompts:
            for pname in all_prompt_names:
                if self.prompts[pname].metadata.original_task:
                    for d in self.dataset:
                        return_value_length = len(self.prompts[pname].apply(d))
                        if return_value_length != 2:
                            invalid_names.append(pname)
                            break
        invalid_names = set(invalid_names)
        prompt_names = [name for name in all_prompt_names if self.prompts[name].metadata.original_task and name not in invalid_names]
        if self.SUBSET_NAME == "cb" and self.cb_surgery:
            return [name for ii, name in enumerate(prompt_names) if ii != 1 and ii != 10]

        return prompt_names


class DatasetByPrompt(Dataset):
    # used for test, maybe need to extend this class for open-ended generation tasks
    def __init__(self, args, cache_dir, tokenizer, split=None, hold_out=-1, random_hold_out=True, testdev_set=False,
                 prompt_names=None):
        super().__init__()
        self.cache_dir = cache_dir

        self.split = split
        self.testdev_set = testdev_set  # if True, always use all prompts for ensemble predictions
        self.DATASET_NAME = args.dataset_name
        self.SUBSET_NAME = args.subset_name if args.subset_name != "none" else None
        self.TESTSET_NAME = args.testset_name
        self.PROMPTSET_NAME = args.prompt_set_name  # has subset name?
        self.task_type = args.task_type
        self.hold_out = hold_out
        self.random_hold_out = random_hold_out  # if False, use the first "hold_out" number of samples in the data
        self.cb_surgery = args.cb_surgery
        self.load()

        self.tokenizer = tokenizer
        self.abl_nprompts = args.abl_nprompts  # for ablation studies
        self.prompts = DatasetTemplates(self.PROMPTSET_NAME, self.SUBSET_NAME)
        if prompt_names is None:
            self.original_task_prompts = self.extract_original_task_prompts(check_valid_prompts=self.SUBSET_NAME=="copa")
        else:
            self.original_task_prompts = prompt_names
        self.construct_meta_info()
        self.num_choices = len(self.prompts[self.original_task_prompts[0]].get_answer_choices_list(self.dataset[0]))
        print("{} has {} original task prompts, number choices = {}, total test examples = {}".format(self.DATASET_NAME + (("/" + self.SUBSET_NAME) if self.SUBSET_NAME is not None else ""),
                                                                                 len(self.original_task_prompts),
                                                                                 self.num_choices,
                                                                                 len(self)))

    def load(self):
        if self.DATASET_NAME == "story_cloze":
            self.dataset = datasets.load_dataset("csv", data_files=os.path.join(self.cache_dir, "cloze_2016_val.csv"))["train"]
        else:
            self.dataset = datasets.load_dataset(self.DATASET_NAME, self.SUBSET_NAME,
                                             cache_dir=self.cache_dir)[self.TESTSET_NAME if self.split is None else self.split]

        if self.hold_out > -1 and len(self.dataset) > self.hold_out:
            selected_data = np.random.choice(len(self.dataset), self.hold_out, replace=False) if self.random_hold_out \
                else range(self.hold_out)
            self.dataset = [self.dataset[int(sidx)] for sidx in selected_data]

    @property
    def num_prompts(self):
        return len(self.original_task_prompts)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        inputs = []
        outputs = []
        prev_label = None
        for pname in self.original_task_prompts:
            input_template, output_template = self.prompts[pname].apply(item)
            if self.task_type == "classification":
                # is output_template always the answer_choices[label]
                targets = self.prompts[pname].get_answer_choices_list(item)
                label = targets.index(output_template.strip())
                assert prev_label is None or label == prev_label
                prev_label = label
                for answer in targets:
                    inputs.append(input_template.strip())
                    outputs.append(answer.strip())
                self.set_num_choices(len(targets))
                # label = self.prompts[pname].get_answer_choices_list(item).index(output_template)
            else:
                # todo: for generation
                pass

        # return inputs, outputs, item['label']
        if self.SUBSET_NAME == "cb" and self.cb_surgery:
            model_inputs = self.tokenizer(inputs, padding=False, truncation=True)
        else:
            model_inputs = self.tokenizer(inputs, padding=False, truncation=True, add_special_tokens=False)
        outputs_ids = self.tokenizer(outputs,  padding=False, truncation=True).input_ids
        model_inputs['labels'] = [[l if l != self.tokenizer.pad_token_id else -100 for l in x] for x in outputs_ids]

        results = [{
            'input_ids': input_id,
            'attention_mask': amask,
            'labels': label,
        } for input_id, amask, label in zip(model_inputs['input_ids'], model_inputs['attention_mask'], model_inputs['labels'])]
        return results, label

    def construct_meta_info(self):
        answer_groups = defaultdict(list)

        for pidx, pname in enumerate(self.original_task_prompts):
            answer_choices = self.prompts[pname].get_fixed_answer_choices_list()
            answer_choices = "_".join(answer_choices) if answer_choices is not None else None
            answer_groups[answer_choices].append(pidx)
        self.prompt_groups = [answer_groups[key] for key in answer_groups.keys()]

    def set_num_choices(self, n):
        check = True if self.num_choices == -1 else self.num_choices == n
        assert check
        self.num_choices = n

    def extract_original_task_prompts(self, check_valid_prompts=False):
        all_prompt_names = self.prompts.all_template_names
        invalid_names = []
        if check_valid_prompts:
            for pname in all_prompt_names:
                if self.prompts[pname].metadata.original_task:
                    for d in self.dataset:
                        return_value_length = len(self.prompts[pname].apply(d))
                        if return_value_length != 2:
                            invalid_names.append(pname)
                            break
        invalid_names = set(invalid_names)
        prompt_names = [name for name in all_prompt_names if self.prompts[name].metadata.original_task and name not in invalid_names]
        if self.SUBSET_NAME == "cb" and self.cb_surgery:
            return [name for ii, name in enumerate(prompt_names) if ii != 1 and ii != 10]

        if self.abl_nprompts > 0 and not self.testdev_set:
            prompt_names = np.random.choice(prompt_names, self.abl_nprompts, replace=False)
        return prompt_names


class TTTOnlineDataset(Dataset):
    def __init__(self, test_dataset, test_args, idx=-1):
        super().__init__()
        train_data_form = test_args.train_data_source
        assert train_data_form == 'stream'
        assert idx >= 0
        self.dataset, self.gold_label = test_dataset[idx]

        self.num_choices = test_dataset.num_choices
        self.num_prompts = test_dataset.num_prompts
        self.original_task_prompts = test_dataset.original_task_prompts

    def __getitem__(self, idx):
        return self.dataset[idx * self.num_choices: (idx+1) * self.num_choices]

    def __len__(self):
        return self.num_prompts

    @property
    def num_examples(self):
        return self.num_prompts * self.num_choices


class TTTOfflineDataset(Dataset):
    def __init__(self, test_dataset, test_args, random_n_prompts):
        super().__init__()
        print("Building TTT training set: {}!".format(test_args.train_data_source))
        train_data_form = test_args.train_data_source
        assert train_data_form != 'stream'

        self.random_n_prompts = random_n_prompts

        self.dataset, self.gold_labels = self.construct_dataset(test_dataset)
        self.datasize = len(test_dataset)

        self.num_choices = test_dataset.num_choices
        self.num_prompts = test_dataset.num_prompts
        self.tot_single_ds_size = self.num_prompts * self.num_choices
        self.original_task_prompts = test_dataset.original_task_prompts

    def construct_dataset(self, dataset: DatasetByPrompt):
        all_data = []
        labels = []
        for examples, label in dataset:
            all_data.extend(examples)
            labels.append(label)
        return all_data, labels

    def __getitem__(self, idx):
        random_prompts = np.random.choice(self.num_prompts, self.random_n_prompts, replace=False)
        results = []
        s = idx * self.tot_single_ds_size
        for rp in random_prompts:
            results.extend(self.dataset[s+rp*self.num_choices : s+(rp+1)*self.num_choices])
        return results

    def __len__(self):
        return self.datasize


class TTTOnlineTokenLossDataset(Dataset):
    def __init__(self, test_dataset, test_args, idx=-1):
        super().__init__()
        train_data_form = test_args.train_data_source
        assert train_data_form == 'stream'
        assert idx >= 0
        self.dataset, self.gold_label = test_dataset[idx]

        self.num_choices = test_dataset.num_choices
        self.num_prompts = test_dataset.num_prompts
        self.original_task_prompts = test_dataset.original_task_prompts

    def __getitem__(self, idx):
        answer_id = np.random.randint(self.num_choices)
        return self.dataset[idx * self.num_choices + answer_id]

    def __len__(self):
        return self.num_prompts

    @property
    def num_examples(self):
        return self.num_prompts


class TTTOfflineTokenLossDataset(Dataset):
    def __init__(self, test_dataset, test_args, random_n_prompts):
        super().__init__()
        print("Building TTT training set: {}!".format(test_args.train_data_source))
        train_data_form = test_args.train_data_source
        assert train_data_form != 'stream'

        self.random_n_prompts = random_n_prompts

        self.dataset, self.gold_labels = self.construct_dataset(test_dataset)
        self.datasize = len(test_dataset)

        self.num_choices = test_dataset.num_choices
        self.num_prompts = test_dataset.num_prompts
        self.tot_single_ds_size = self.num_prompts * self.num_choices
        self.original_task_prompts = test_dataset.original_task_prompts

    def construct_dataset(self, dataset: DatasetByPrompt):
        all_data = []
        labels = []
        for examples, label in dataset:
            all_data.extend(examples)
            labels.append(label)
        return all_data, labels

    def __getitem__(self, idx):
        random_prompts = np.random.choice(self.num_prompts, self.random_n_prompts, replace=False)
        results = []
        item_idx = idx % self.datasize
        ans_idx = idx // self.datasize
        s = item_idx * self.tot_single_ds_size
        for rp in random_prompts:
            results.append(self.dataset[s+rp*self.num_choices+ans_idx])
        return results

    def __len__(self):
        return self.datasize * self.num_choices


class TTTOfflineLoopDataset(Dataset):
    def __init__(self, test_dataset, test_args, random_n_prompts, dev_bsz, idx=-1):
        super().__init__()
        print("Building TTT training set: {}!".format(test_args.train_data_source))
        train_data_form = test_args.train_data_source
        if train_data_form != 'stream':
            assert idx == -1
            self.dataset, self.gold_labels = self.construct_dataset(test_dataset)
            self.datasize = len(test_dataset)
        else:
            self.dataset, gold_label = test_dataset[idx]
            self.gold_labels = [gold_label]

        self.random_n_prompts = random_n_prompts
        self.dev_bsz = dev_bsz

        self.num_choices = test_dataset.num_choices
        self.num_prompts = test_dataset.num_prompts
        self.tot_single_ds_size = self.num_prompts * self.num_choices
        self.original_task_prompts = test_dataset.original_task_prompts
        self.prompt_groups = test_dataset.prompt_groups

        tot = self.tot_single_ds_size
        self.dev_batches = [(i, i+dev_bsz if i+dev_bsz < tot else tot) for i in range(0, tot, dev_bsz)]
        self.dev_size = len(self.dev_batches)

        self.split_answer_groups = test_args.split_answer_groups

    def construct_dataset(self, dataset: DatasetByPrompt):
        all_data = []
        labels = []
        for examples, label in dataset:
            all_data.extend(examples)
            labels.append(label)
        return all_data, labels

    def __len__(self):
        return self.datasize

    def __getitem__(self, idx):
        results = []
        s = idx * self.tot_single_ds_size
        for bidx1, bidx2 in self.dev_batches:
            results.append(self.dataset[s+bidx1: s+bidx2])

        if self.split_answer_groups:
            for pgroup in self.prompt_groups:
                random_prompts = pgroup if len(pgroup) <= self.random_n_prompts \
                    else np.random.choice(pgroup, self.random_n_prompts, replace=False)
                for ans_idx in range(self.num_choices):
                    results.append([self.dataset[s + pid * self.num_choices + ans_idx] for pid in random_prompts])
        else:
            random_prompts = list(range(self.num_prompts)) if self.num_prompts <= self.random_n_prompts \
                                                            else np.random.choice(self.num_prompts, self.random_n_prompts, replace=False)
            for ans_idx in range(self.num_choices):
                results.append([self.dataset[s + pid * self.num_choices + ans_idx] for pid in random_prompts])
        return results


class TTTEvalDataset(Dataset):
    def __init__(self, test_dataset):
        super().__init__()
        print("Building TTT evaluation test set!")
        self.num_instances = len(test_dataset)
        self.dataset, self.gold_labels = self.construct_dataset(test_dataset)

        self.num_choices = test_dataset.num_choices
        self.num_prompts = test_dataset.num_prompts
        self.original_task_prompts = test_dataset.original_task_prompts

    def construct_dataset(self, dataset: DatasetByPrompt):
        all_data = []
        labels = []
        for examples, label in dataset:
            all_data.extend(examples)
            labels.append(label)
        return all_data, labels

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class TrainDatasetByPrompt(Dataset):
    # used for train data, extended for open-ended generation tasks
    def __init__(self, args, cache_dir, tokenizer, split=None, hold_out=-1, random_hold_out=True):
        super().__init__()
        self.cache_dir = cache_dir

        self.split = split
        self.DATASET_NAME = args.dataset_name
        self.SUBSET_NAME = args.subset_name if args.subset_name != "none" else None
        self.TESTSET_NAME = args.testset_name
        self.PROMPTSET_NAME = args.prompt_set_name  # has subset name?
        self.task_type = args.task_type
        self.hold_out = hold_out
        self.random_hold_out = random_hold_out  # if False, use the first "hold_out" number of samples in the data
        self.load()

        self.tokenizer = tokenizer

        self.prompts = DatasetTemplates(self.PROMPTSET_NAME, self.SUBSET_NAME)
        self.task_prompts, self.original_task_prompts = self.extract_original_task_prompts(check_valid_prompts=False)

        choices = self.prompts[self.original_task_prompts[0]].get_answer_choices_list(self.dataset[0])
        if choices is not None:
            # this might not be true, as different prompts might have different number of choices
            self.num_choices = len(choices)
        else:
            self.num_choices = -1
        print("{} has {} task prompts, number choices = {}, total test examples = {}".format(self.DATASET_NAME + (("/" + self.SUBSET_NAME) if self.SUBSET_NAME is not None else ""),
                                                                                            len(self.original_task_prompts),
                                                                                            self.num_choices,
                                                                                            len(self)))

    def load(self):
        if self.DATASET_NAME == "story_cloze":
            self.dataset = datasets.load_dataset("csv", data_files=os.path.join(self.cache_dir, "cloze_2016_val.csv"))["train"]
        else:
            self.dataset = datasets.load_dataset(self.DATASET_NAME, self.SUBSET_NAME,
                                             cache_dir=self.cache_dir)[self.TESTSET_NAME if self.split is None else self.split]

        if self.hold_out > -1 and len(self.dataset) > self.hold_out:
            selected_data = np.random.choice(len(self.dataset), self.hold_out, replace=False) if self.random_hold_out \
                else range(self.hold_out)
            self.dataset = [self.dataset[int(sidx)] for sidx in selected_data]

    @property
    def num_prompts(self):
        return len(self.task_prompts)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        inputs = []
        outputs = []
        prompt_groups = defaultdict(list) # put in the prompt name and its position in the inputs
        prev_label = None
        for pname in self.task_prompts:
            if len(self.prompts[pname].apply(item)) != 2:
                continue
            input_template, output_template = self.prompts[pname].apply(item)
            choices = self.prompts[pname].get_answer_choices_list(item)
            if choices is not None:
                num_choices = len(choices)
                label = choices.index(output_template.strip())
                prompt_groups["{}_{}".format(num_choices, label)].append((pname, len(inputs), input_template.strip(), output_template.strip(), choices))
                for answer in choices:
                    inputs.append(input_template.strip())
                    outputs.append(answer.strip())
            else:
                prompt_groups["1_{}".format(output_template.strip())].append((pname, len(inputs), input_template.strip(), output_template.strip(), None))
                inputs.append(input_template.strip())
                outputs.append(output_template.strip())
                # return inputs, outputs, item['label']
        if self.SUBSET_NAME == "cb" and self.cb_surgery:
            model_inputs = self.tokenizer(inputs, padding=False, truncation=True)
        else:
            model_inputs = self.tokenizer(inputs, padding=False, truncation=True, add_special_tokens=False)
        outputs_ids = self.tokenizer(outputs,  padding=False, truncation=True).input_ids
        model_inputs['labels'] = [[l if l != self.tokenizer.pad_token_id else -100 for l in x] for x in outputs_ids]

        results = [{
            'input_ids': input_id,
            'attention_mask': amask,
            'labels': label,
        } for input_id, amask, label in zip(model_inputs['input_ids'], model_inputs['attention_mask'], model_inputs['labels'])]
        return results, prompt_groups

    def extract_original_task_prompts(self, check_valid_prompts=False):
        all_prompt_names = self.prompts.all_template_names
        invalid_names = []
        if check_valid_prompts:
            for pname in all_prompt_names:
                if self.prompts[pname].metadata.original_task:
                    for d in self.dataset:
                        return_value_length = len(self.prompts[pname].apply(d))
                        if return_value_length != 2:
                            invalid_names.append(pname)
                            break
        invalid_names = set(invalid_names)
        original_prompt_names = [name for name in all_prompt_names if self.prompts[name].metadata.original_task and name not in invalid_names]
        prompt_names = [name for name in all_prompt_names if name not in invalid_names]
        if self.SUBSET_NAME == "cb" and self.cb_surgery:
            return [name for ii, name in enumerate(prompt_names) if ii != 1 and ii != 10]
        return prompt_names, original_prompt_names
