import ast

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from datasets import load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)
from transformers.pipelines.pt_utils import KeyDataset


def main():
    # model_name = "cheongb002/Mistral-Planner-7B_new_data"
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model_name = "/home/brian/Documents/School/ECE1724/P3/GPT-Driver/results-Mistral-Planner-7B_new_data/checkpoint-4000"
    dataset_name = "/home/brian/Documents/School/ECE1724/P3/GPT-Driver/data/av2_mistral"

    dataset = load_from_disk(dataset_name)["val"].shuffle(
        seed=42).select(range(100))

    compute_dtype = getattr(torch, 'float16')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, batch_size=4)

    pos_err = []
    vel_err = []
    heading_err = []
    broken_responses = []
    dataset_pruned = dataset.map(lambda x: {"messages": x["messages"][:1]})
    dataset_pruned = dataset_pruned.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(
        x["messages"], tokenize=False, add_generation_prompt=True)})
    responses = []
    for response in tqdm.tqdm(generator(KeyDataset(dataset_pruned, key="formatted_chat"), max_new_tokens=64)):
        response = response[0]["generated_text"]
        responses.append(response)

    for i, (data, response) in enumerate(zip(dataset["messages"], responses)):
        response_split = response.split(
            "[/INST]")[-1].split("\n")
        try:
            pred_target_pos = ast.literal_eval(
                " ".join(response_split[0].split(" ")[-2:]))
            pred_target_vel = ast.literal_eval(
                " ".join(response_split[1].split(" ")[-2:]))
            pred_target_heading = ast.literal_eval(
                response_split[2].split(" ")[-1])
        except:
            broken_responses.append(response)
            continue
        gt_target = data[-1]["content"].split("\n")
        gt_target_pos = ast.literal_eval(
            " ".join(gt_target[0].split(" ")[-2:]))
        gt_target_vel = ast.literal_eval(
            " ".join(gt_target[1].split(" ")[-2:]))
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

    print("Average Position Error: ", np.mean(pos_err))
    plt.subplot(1, 3, 1)
    plt.hist(pos_err, bins=20)
    plt.xlabel('Position Error')
    plt.ylabel('Frequency')

    print("Average Velocity Error: ", np.mean(vel_err))
    plt.subplot(1, 3, 2)
    plt.hist(vel_err, bins=20)
    plt.xlabel('Velocity Error')
    plt.ylabel('Frequency')

    print("Average Heading Error: ", np.mean(heading_err))
    plt.subplot(1, 3, 3)
    plt.hist(heading_err, bins=20)
    plt.xlabel('Heading Error')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
