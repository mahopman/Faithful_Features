import goodfire
from dotenv import load_dotenv
import argparse

from dataset_curation import (
    load_and_format_mmlu_dataset,
    create_dataset,
    remove_incorrect_answers,
    save_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Process MMLU dataset.")
    parser.add_argument(
        "--filename",
        type=str,
        default="data/reasoning.csv",
        help="Output filename for the dataset",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="meta-llama/Meta-Llama-3.1-70B-Instruct",
        help="Goodfire model variant to use for reasoning.",
    )
    parser.add_argument(
        "--dataset_size",
        type=str,
        default=10,
        help="The number of MMLU questions to process.",
    )
    args = parser.parse_args()

    load_dotenv()
    variant = goodfire.Variant(args.variant)
    print("Loading dataset...")
    sample_df = load_and_format_mmlu_dataset(args.dataset_size)
    print("Gathering reasoning and answers...")
    dataset = create_dataset(sample_df, variant)
    print(dataset)
    print("Removing incorrect answers...")
    dataset = remove_incorrect_answers(dataset)
    print("Saving dataset...")
    save_dataset(dataset, args.filename)
    print("Done!")


if __name__ == "__main__":
    main()
