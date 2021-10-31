from promptsource.templates import DatasetTemplates
import datasets


class Task():
    # used for test, maybe need to extend this class for open-ended generation tasks
    def __init__(self, args, cache_dir):
        self.cache_dir = cache_dir
        self.DATASET_NAME = args.dataset_name
        self.SUBSET_NAME = args.subset_name if args.subset_name != "none" else None
        self.TESTSET_NAME = args.testset_name
        self.PROMPTSET_NAME = args.prompt_set_name  # has subset name?
        self.task_type = args.task_type
        self.load()

        self.prompts = DatasetTemplates(self.PROMPTSET_NAME, self.SUBSET_NAME)
        self.original_task_prompts = self.extract_original_task_prompts()
        print("{} has {} original task prompts, total test examples = {}".format(self.DATASET_NAME + ("/" + self.SUBSET_NAME) if self.SUBSET_NAME is not None else "",
                                                                                 len(self.original_task_prompts),
                                                                                 self.size))

    def load(self):
        self.data = datasets.load_dataset(self.DATASET_NAME, self.SUBSET_NAME, cache_dir=self.cache_dir)[self.TESTSET_NAME]

    @property
    def num_prompts(self):
        return len(self.original_task_prompts)

    @property
    def size(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = []
        outputs = []
        prev_label = None
        for pname in self.original_task_prompts:
            input_template, output_template = self.prompts[pname].apply(item)
            if self.task_type == "classification":
                # fixme: for each pname, enumerate all possible answer choices.
                # is output_template always the answer_choices[label]
                all_outputs = []
                all_inputs = []
                for answer in self.prompts[pname].get_answer_choices_list(item):
                    all_inputs.append(input_template)
                    all_outputs.append(answer)
                label = self.prompts[pname].get_answer_choices_list(item).index(output_template)
                check = True if prev_label is None else label == prev_label
                prev_label = label
                assert check
                inputs.append(all_inputs)
                outputs.append(all_outputs)
            else:
                inputs.append([input_template])
                outputs.append([output_template])
        return inputs, outputs, prev_label

    def extract_original_task_prompts(self):
        all_prompt_names = self.prompts.all_template_names
        return [name for name in all_prompt_names if self.prompts[name].metadata.original_task]