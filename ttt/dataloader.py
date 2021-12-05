from promptsource.templates import DatasetTemplates
import datasets
from torch.utils.data import Dataset
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DatasetByPrompt(Dataset):
    # used for test, maybe need to extend this class for open-ended generation tasks
    def __init__(self, args, cache_dir, tokenizer, split=None):
        super().__init__()
        self.cache_dir = cache_dir

        self.split = split
        self.DATASET_NAME = args.dataset_name
        self.SUBSET_NAME = args.subset_name if args.subset_name != "none" else None
        self.TESTSET_NAME = args.testset_name
        self.PROMPTSET_NAME = args.prompt_set_name  # has subset name?
        self.task_type = args.task_type
        self.load()

        self.tokenizer = tokenizer
        self.prompts = DatasetTemplates(self.PROMPTSET_NAME, self.SUBSET_NAME)
        self.original_task_prompts = self.extract_original_task_prompts()
        self.num_choices = len(self.prompts[self.original_task_prompts[0]].get_answer_choices_list(self.dataset[0]))
        print("{} has {} original task prompts, number choices = {}, total test examples = {}".format(self.DATASET_NAME + ("/" + self.SUBSET_NAME) if self.SUBSET_NAME is not None else "",
                                                                                 len(self.original_task_prompts),
                                                                                 self.num_choices,
                                                                                 len(self)))

    def load(self):
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
        for pname in self.original_task_prompts:
            input_template, output_template = self.prompts[pname].apply(item)
            if self.task_type == "classification":
                # is output_template always the answer_choices[label]
                targets = self.prompts[pname].get_answer_choices_list(item)
                for answer in targets:
                    inputs.append(input_template)
                    outputs.append(answer.strip())
                self.set_num_choices(len(targets))
                # label = self.prompts[pname].get_answer_choices_list(item).index(output_template)
            else:
                # todo: for generation
                inputs.append([input_template])
                outputs.append([output_template.strip()])

        # return inputs, outputs, item['label']
        model_inputs = self.tokenizer(inputs, padding=False, truncation=True)
        outputs_ids = self.tokenizer(outputs,  padding=False, truncation=True).input_ids
        model_inputs['labels'] = [[l if l != self.tokenizer.pad_token_id else -100 for l in x] for x in outputs_ids]

        results = [{
            'input_ids': input_id,
            'attention_mask': amask,
            'labels': label,
        } for input_id, amask, label in zip(model_inputs['input_ids'], model_inputs['attention_mask'], model_inputs['labels'])]
        return results, item['label']

    def set_num_choices(self, n):
        check = True if self.num_choices == -1 else self.num_choices == n
        assert check
        self.num_choices = n

    def extract_original_task_prompts(self):
        all_prompt_names = self.prompts.all_template_names
        return [name for name in all_prompt_names if self.prompts[name].metadata.original_task]


class TTTDataset(Dataset):
    def __init__(self, test_dataset, test_args, random_n_prompts, idx=-1):
        super().__init__()
        train_data_form = test_args.train_data_source
        assert train_data_form == 'stream'
        assert idx >= 0
        self.dataset, self.gold_label = test_dataset[idx]

        self.num_choices = test_dataset.num_choices
        self.num_prompts = test_dataset.num_prompts
        self.random_n_prompts = random_n_prompts
        self.original_task_prompts = test_dataset.original_task_prompts

    def __getitem__(self, idx):
        # return self.dataset[idx * self.num_choices: (idx+1) * self.num_choices]
        random_prompts = np.random.choice(self.num_prompts, self.random_n_prompts, replace=False)
        results = []
        # s = idx * self.tot_single_ds_size
        for rp in random_prompts:
            results.extend(self.dataset[rp*self.num_choices : (rp+1)*self.num_choices])
        return results

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
