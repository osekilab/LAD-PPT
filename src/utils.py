import os
import random
from collections import defaultdict

import datasets
import fire
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm, trange
from transformers import AutoTokenizer


MAX_LENGTH = 1024
DEFAULT_TOKENIZER = "EleutherAI/pythia-1b"


def load_text_files(file_dir):
    texts = []
    for filename in tqdm(os.listdir(file_dir)):
        if filename.endswith(".txt"):
            file_path = os.path.join(file_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return datasets.Dataset.from_dict({"text": texts})


def cache_data_txt(
    file_dir: str, out_dir: str, tokenizer_name: str = DEFAULT_TOKENIZER
):
    dataset = load_text_files(file_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH
        ),
        batched=True,
    )

    dataset.save_to_disk(out_dir)


def cache_data(
    dataset_name: str = None,
    out_dir: str = None,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    data_files: str = None,
    max_length: int = MAX_LENGTH,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    if data_files is not None:
        files_list = [f.strip() for f in data_files.split(",")]
        dataset = load_dataset("json", data_files=files_list, split="train")
        dataset = dataset.map(
            lambda x: tokenizer(
                x["text"],
                truncation=True,
                max_length=max_length,
            ),
        ).remove_columns(["text"])
    elif dataset_name is not None:
        if dataset_name == "blimp":
            dataset = datasets.load_dataset("WillHeld/blimp", split="train")

            def tokenize_examples(examples):
                good = tokenizer(
                    examples["sentence_good"],
                    truncation=True,
                    max_length=128,
                )
                bad = tokenizer(
                    examples["sentence_bad"],
                    truncation=True,
                    max_length=128,
                )
                return {
                    "good_input_ids": good["input_ids"],
                    "good_attention_mask": good["attention_mask"],
                    "bad_input_ids": bad["input_ids"],
                    "bad_attention_mask": bad["attention_mask"],
                }

            cols = dataset.column_names
            dataset = dataset.map(tokenize_examples, batched=True).remove_columns(cols)
        elif dataset_name == "wikitext":
            dataset = load_dataset(
                "Salesforce/wikitext", name="wikitext-2-v1", split="train"
            )
            dataset = dataset.map(
                lambda x: tokenizer(x["text"], truncation=True, max_length=max_length),
                batched=True,
            )
        elif "/" in dataset_name and not os.path.exists(dataset_name):
            dataset = load_dataset(dataset_name, "en", split="train")
            dataset = dataset.map(
                lambda x: tokenizer(
                    x["text"],
                    truncation=True,
                    max_length=max_length,
                ),
                batched=True,
            ).remove_columns(["text"])
        else:
            dataset = load_dataset("text", data_files=dataset_name, split="train")
            dataset = dataset.map(
                lambda x: tokenizer(
                    x["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                ),
            ).remove_columns(["text"])
    else:
        raise ValueError("Either dataset_name or data_files must be provided")

    dataset.save_to_disk(out_dir)


def cache_c4_local(
    out_dir: str,
    local_dir: str,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    num_proc: int = 8,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    print(f"Loading C4 from: {local_dir}")
    dataset = load_from_disk(local_dir)
    if "train" in dataset:
        dataset = dataset["train"]

    print(f"Loaded {len(dataset)} examples. Tokenizing using {num_proc} processes...")
    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            max_length=MAX_LENGTH
        ),
        batched=True,
        num_proc=num_proc
    ).remove_columns(["text"])

    os.makedirs(out_dir, exist_ok=True)
    dataset.save_to_disk(out_dir)
    print(f"Tokenized dataset saved to: {out_dir}")


if __name__ == "__main__":
    fire.Fire(
        {
            "cache_data_txt": cache_data_txt,
            "cache_data": cache_data,
            "cache_c4_local": cache_c4_local,
        }
    )
