# Image classification

## Index
- [Environment setup](#environment-setup)
- [Running the code](#running-the-code)
- [Configuration](#configuration)
- [Acknowledgements](#acknowledgements)


## Environment setup
First, navigate to this directory:
```bash
cd rw2s/vision
```
Then, set up an environment from the `environment.yaml` file and activate it ([Miniconda](https://docs.anaconda.com/free/miniconda/index.html)):
```bash
conda env create -f environment.yaml
conda activate rw2s
```

Finally, install the local `rw2s` package:
```bash
pip install -e .
```


## Running the code
The scripts for running the full pipeline on vision tasks (`rw2s/vision/run_w2s.py` and `rw2s/vision/cosupervised_learning/run_csl.py`) require a single argument `cfg_path`: the path to the configuration file. For example:
```bash
python run_w2s.py --cfg_path ../configs/fmow.yaml
```
or
```bash
python cosupervised_learning/run_csl.py --cfg_path ../configs/csl/csl_fmow.yaml
```
for co-supervised learning (CSL).


## Configuration

Provided configuration files can be used for both single-weak-model approaches (e.g., *Conf*, *V-sup*, etc.), as well as ensemble-based methods (e.g., *RAVEN*, *Ens*, etc.). The configuration files are in YAML format and are placed in the `rw2s/configs` directory. Each configuration file is named as `{dataset}.yaml` (e.g. `iwildcam.yaml`, `fmow.yaml`, etc.). **Make sure to set the correct file paths according to your local setup.**

Specific configuration parameters can be found as comments in the configuration files. For *RAVEN*, the important parameters include `ensemble.ensemble_weights_opter_cfg` (adaptive weighting and easy-sample guided initialization) and `w2s.ignore_samples_with_disagreement_first_n_epochs` (easy-sample guided initialization). The current (default) configuration parameters are set for the *Naive* baseline.

**Weak model training.** The configuration files are setup such that one can first train the weak models one by one, and later compose them in an ensemble for weak-to-strong training (list multiple weak models in `ensemble.weak_models` with `load_ckpt` set to a pre-trained checkpoint). To train a weak model, set `train: true` for some weak model in the `ensemble.weak_models` list, and specify the training configuration `train_cfg`.

**Co-supervised learning.** Weak models for co-supervised learning (CSL) should be first trained using the configuration files in `rw2s/configs` (setting `domain_start_end_idxs` and `domain_name` in training configuration of weak models), and their soft labels should be saved along with the strong model's embeddings using the `w2s.save_labels: true` and `w2s.save_embeddings: true` options. Then, the saved soft labels of multiple weak models and strong-model embeddings can be used in the CSL pipeline by specifying the soft label and embedding paths in `rw2s/configs/csl` configuration files.


## Acknowledgements
- A large part of the codebase is based on the [repository of the original weak-to-strong paper](https://github.com/openai/weak-to-strong.git).
- The (modified) code for co-supervised learning is based on [this repository](https://github.com/YuejiangLIU/csl.git).
