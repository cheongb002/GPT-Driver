import json
import os
import pickle

import tqdm
from datasets import Dataset
from prompt_message import (generate_assistant_message, generate_user_message,
                            system_message)


def format_conversation(row):
    template = "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"

    conversation = template.format(
        sys=system_message,
        q=row["user"],
        a=row["assistant"],
    )

    return {"text": conversation}


data = pickle.load(open('data/cached_nuscenes_info.pkl', 'rb'))
split = json.load(open('data/split.json', 'r'))

train_tokens = split["train"]
val_tokens = split["val"]
num_train_samples = len(train_tokens)
train_ratio = 1


num_language_tokens = 0
num_system_tokens = 0
num_user_tokens = 0
num_assistant_tokens = 0

traj_only = False

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
    # if len(assitant_message.split("\n")) > 6:
    #     print()
    #     print(token)
    #     print(system_message)
    #     print(user_message)
    #     print(assitant_message)

    # train_message = {"messages":
    #     [
    #         {"role": "system", "content": system_message},
    #         {"role": "user", "content": user_message},
    #         {"role": "assistant", "content": assitant_message}
    #     ]
    # }
    # train_messages.append(train_message)

# with open("data/train.json", "w") as f:
#     ndjson.dump(train_messages, f)
dataset = Dataset.from_list(train_messages)
dataset = dataset.map(
    format_conversation,
    # remove all columns; only "text" will be left
    remove_columns=dataset.column_names,
    num_proc=os.cpu_count()  # multithreaded
)
dataset.save_to_disk("data/processed")
