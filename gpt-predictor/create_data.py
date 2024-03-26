import json
import os
import pickle

import tqdm
from prompt_message import (generate_assistant_message, generate_user_message,
                            system_message)

from datamodules import ArgoverseV2DataModule


def format_conversation(row):
    template = "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"

    conversation = template.format(
        sys=system_message,
        q=row["user"],
        a=row["assistant"],
    )

    return {"text": conversation}


# Construct dataset
datamodule = ArgoverseV2DataModule(
    root="data/av2",
    train_batch_size=1,
    val_batch_size=1,
    test_batch_size=1,
    shuffle=False,
    num_workers=8,
)

train_dataloader, val_dataloader, test_dataloader = datamodule.train_dataloader(
), datamodule.val_dataloader(), datamodule.test_dataloader()

# process train dataloader
for batch in tqdm.tqdm(train_dataloader):
    breakpoint()

# assistant message: 

train_messages = []
for token_i, token in tqdm.tqdm(enumerate(train_tokens)):
    if token_i >= train_ratio * num_train_samples:
        break
    user_message = generate_user_message(data, token)
    assitant_message = generate_assistant_message(
        data, token, traj_only=traj_only)
    train_messages.append({
        "user": user_message,
        "assistant": assitant_message
    })

dataset = Dataset.from_list(train_messages)
dataset = dataset.map(
    format_conversation,
    # remove all columns; only "text" will be left
    remove_columns=dataset.column_names,
    num_proc=os.cpu_count()  # multithreaded
)
dataset.save_to_disk("data/processed")
