# Prompt Consistency for Zero-Shot Task Generalization
This is the official implementation of the [paper](https://arxiv.org/abs/2205.00049):

```
Prompt Consistency for Zero-Shot Task Generalization
Chunting Zhou*, Junxian He*, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig
Preprint 2022
```

## Dependencies

This repo is a fork of the [huggingface transformers](https://github.com/huggingface/transformers) repo (forked on Oct 29, 2021), and the code is tested on [PyTorch](https://pytorch.org) 1.9.0. Please follow the instructions below to install dependencies after you set up PyTorch:

```bash
git clone git@github.com:violet-zct/swarm-distillation-zero-shot.git
cd swarm-distillation-zero-shot

# install transformers from this repo
pip install -e .

# install other requirements
pip install datasets
```

## Usage
We are still working on cleaning the code, for early usage please refer to `exps/ttt/final_3B.sh` for an example training script that we used to tune the T0-3B model.

## Citation

```
@article{zhou2022prompt,
  title={Prompt Consistency for Zero-Shot Task Generalization},
  author={Chunting Zhou and Junxian He and Xuezhe Ma and Taylor Berg-Kirkpatrick and Graham Neubig},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.00049}
}
```