# [RLHF-Reward-Modeling: BT Model](https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/bradley-terry-rm)

## Installation instructions

The current solution is based on the alignment handbook and the environment, which should be sufficient for plain RM training.
Before starting, please make sure your linux machine has [nvidia-cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) installed.

```shell
conda create -n rm_dev python=3.10.9
conda activate rm_dev

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout d17fd7cd3b71c6a7bf7af34d8dc73135bb7ea8e9

# The test cuda version is 12.1, 12.2. You may need to update the torch version based on your cuda version...
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .

pip install flash-attn==2.6.3

pip install accelerate==0.33.0 # for gemma2 and llama3.1
pip install deepspeed==0.12.2
pip install transformers==4.43.4
pip install numpy==1.26.4 # Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!


cd robust-weak-to-strong-generalization/rw2s/DPO
```

You also need to install wandb to record the training and log in with the huggingface accout to access Gemma.

```shell
pip install wandb
wandb login

huggingface-cli login
```

Some possible problems:

`CUDA_HOME` may not exist, unable to compile CUDA op(s)AssertionError:[end of output]

```shell
conda install nvidia/label/cuda-12.2.0::cuda-nvcc
```

## Dataset Preparation
The dataset should be preprocessed as the standard format, where each of the sample consists of two conversations 'chosen' and 'rejected' and they share the same prompt. Here is an example of the rejected sample in the comparison pair. 

```python
[
{ "content": "Please identify the top 5 rarest animals in the world.", "role": "user" },
{ "content": "Do you mean animals that are really rare, or rare relative to the size of the human population?", "role": "assistant" },
{ "content": "The ones that are really rare.", "role": "user" },
{ "content": "Alright, here’s what I found:", "role": "assistant" }, 
]
```

We used the preprocessed open-source preference datasets [HERE](https://huggingface.co/collections/RLHFlow/standard-format-preference-dataset-662eec0252e194d5d40c252a), including [Harmless](https://huggingface.co/datasets/RLHFlow/HH-RLHF-Harmless-and-RedTeam-standard) and [Helpfulness](https://huggingface.co/datasets/RLHFlow/HH-RLHF-Helpful-standard) subsets of HH-RLHF.

## Running the Code

Running the code with Llama-3.2-1B with train.sh

```shell
epochs=5
dataset=RLHFlow/HH-RLHF-Helpful-standard
target_dataset=RLHFlow/HH-RLHF-Harmless-and-RedTeam-standard 
model=meta-llama/Llama-3.2-1B
lr=0.00001
port=25905
outpath=./bt_models_${epochs}/help_llama_$lr
accelerate launch --main_process_port $port ./bradley-terry-rm/rm.py --model_name $model --max_length 4096 --train_set_path $dataset --output_path $outpath --target_set_path $target_dataset --learning_rate $lr --num_train_epochs $epochs &
```
