from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import datasets
from datasets import load_dataset
import os
from tqdm import tqdm
import fire

def llm_infer(
    model_path: str,
    model_name: str = "Qwen/Qwen2.5-7B",
    dataset: str = "harmless_gt",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    max_new_tokens: int = 512,
    batch_size: int = 1
    #repetition_penalty: float = 1.0,
):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side='left'
    data = load_dataset(dataset)
    test = data['test'].shuffle(seed=1)
    resume=0    
    mode = dataset.split('_')[0]
    if os.path.exists(save_name):
        with open(save_name, "r") as f:
            saved = [json.loads(line) for line in f]
        resume=len(saved)
    out=[]

    for i in tqdm(range(0,len(test), batch_size)): #for i, chat in enumerate(tqdm(test)):
        if i < resume:
            continue
        batch = test[i:i+batch_size]
        texts=[]
        prompts = batch['instruction']
        histories = batch['history']
        chosens=batch['chosen']
        for inst in zip(prompts,histories):
            prompt, hist_inst = inst
            messages = []
            history = [{"role": "system", "content": f" You are a {mode} assistant."}]
            for hist in hist_inst:
                history.extend([
                    {"role": "user", "content": hist[0]},
                    {"role": "assistant", "content": hist[1]}])

            messages.extend(history)
            messages.extend([
                {"role": "user", "content": prompt}
            ])
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for out in zip(prompts, responses, chosens, histories):
            prompt, response, chosen, history = out
            with open(save_name,'a', encoding="utf-8") as f:
                f.write(json.dumps({"prompt": prompt, "predict": response, "label": chosen, "history": history }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    fire.Fire(llm_infer)
