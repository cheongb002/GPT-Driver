import ast

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from create_data import system_message
from datasets import load_from_disk
from peft import LoraConfig, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)
from trl import setup_chat_format

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
dataset_name = "/home/brian/Documents/School/ECE1724/P3/GPT-Driver/data/av2_conversational"
checkpoint = "/home/brian/Documents/School/ECE1724/P3/GPT-Driver/results_mistral/checkpoint-5000"

dataset = load_from_disk(dataset_name)["val"].shuffle(
    seed=42).select(range(100))

compute_dtype = getattr(torch, 'float16')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

model, tokenizer = setup_chat_format(model, tokenizer)
model = PeftModel.from_pretrained(model, checkpoint)

generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, batch_size=4)

pos_err = []
vel_err = []
heading_err = []
broken_responses = []
for data in tqdm.tqdm(dataset["messages"]):
    assert data[2]["role"] == "assistant", "Assistant should be the third message"
    data_pruned = data[:2]
    data_pruned = tokenizer.apply_chat_template(
        data_pruned, tokenize=False, add_generation_prompt=True)
    response = generator(data_pruned, max_new_tokens=51)[0]["generated_text"]
    response_split = response.split("\n")[-3:]
    try:
        pred_target_pos = ast.literal_eval(" ".join(response_split[0].split(" ")[-2:]))
        pred_target_vel = ast.literal_eval(" ".join(response_split[1].split(" ")[-2:]))
        pred_target_heading = ast.literal_eval(response_split[2].split(" ")[-1])
    except:
        broken_responses.append(response)
        continue

    gt_target = data[2]["content"].split("\n")
    gt_target_pos = ast.literal_eval(" ".join(gt_target[0].split(" ")[-2:]))
    gt_target_vel = ast.literal_eval(" ".join(gt_target[1].split(" ")[-2:]))
    gt_target_heading = ast.literal_eval(gt_target[2].split(" ")[-1])
    pos_err.append(np.linalg.norm(
        np.array(pred_target_pos) - np.array(gt_target_pos)))
    vel_err.append(np.linalg.norm(
        np.array(pred_target_vel) - np.array(gt_target_vel)))
    heading_err.append(
        np.abs(np.array(pred_target_heading) - np.array(gt_target_heading)))


# Normalize heading errors to [0, pi]
heading_err = np.mod(heading_err, np.pi)

print("broken responses", broken_responses)
# Create histograms
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(pos_err, bins=20)
plt.xlabel('Position Error')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(vel_err, bins=20)
plt.xlabel('Velocity Error')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.hist(heading_err, bins=20)
plt.xlabel('Heading Error')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()