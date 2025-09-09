#dataset=RLHFlow/HH-RLHF-Harmless-and-RedTeam-standard 
#accelerate launch ./bradley-terry-rm/qwen25_05B_rm.py --max_length 4096 --train_set_path $dataset --lora --output_path $outpath&
epochs=5
dataset=RLHFlow/HH-RLHF-Helpful-standard
target_dataset=RLHFlow/HH-RLHF-Harmless-and-RedTeam-standard 
model=meta-llama/Llama-3.2-1B
lr=0.00001
port=25905
outpath=./bt_models_${epochs}/help_llama_$lr
accelerate launch --main_process_port $port ./bradley-terry-rm/rm.py --model_name $model --max_length 4096 --train_set_path $dataset --output_path $outpath --target_set_path $target_dataset --learning_rate $lr --num_train_epochs $epochs &&
