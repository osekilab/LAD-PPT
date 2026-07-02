import os
import fire
from datasets import load_dataset


def download_c4_hub(
    out_dir: str = "./data/c4_en_raw",
    config_name: str = "en",
):
    print(f"Loading full C4 ('{config_name}' config) via HuggingFace datasets API.")
    print("Note: This requires `huggingface-cli login` and access approval for allenai/c4.")

    dataset = load_dataset(
        "allenai/c4",
        config_name,
        split="train",
    )

    dataset.save_to_disk(out_dir)
    print(f"C4 dataset saved to: {out_dir}")
    print(f"Dataset size: {len(dataset)} examples.")


def download_c4_subset(
    out_dir: str = "./data/c4_en_subset",
    config_name: str = "en",
    num_examples: int = 5000000,
):
    print(f"Loading C4 ('{config_name}' config) via streaming, taking {num_examples} examples...")
    print("Note: This requires `huggingface-cli login` and access approval for allenai/c4.")

    streaming_dataset = load_dataset(
        "allenai/c4",
        config_name,
        split="train",
        streaming=True,
    )

    subset = list(streaming_dataset.take(num_examples))
    from datasets import Dataset
    dataset = Dataset.from_list(subset)

    dataset.save_to_disk(out_dir)
    print(f"C4 subset ({len(dataset)} examples) saved to: {out_dir}")


if __name__ == "__main__":
    fire.Fire(
        {
            "download_hub": download_c4_hub,
            "download_subset": download_c4_subset,
        }
    )
