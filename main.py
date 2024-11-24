import goodfire
from dotenv import load_dotenv
import argparse
import pandas as pd
import os

from dataset_curation import (
    load_and_format_mmlu_dataset,
    create_dataset,
    remove_incorrect_answers,
    save_dataset,
)

from experiment_generation import search_features, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Process MMLU dataset.")
    parser.add_argument(
        "--dataset_filename",
        type=str,
        default="data/reasoning.csv",
        help="Output filename for the dataset. If the file does not exist, a new dataset will be created.",
    )
    parser.add_argument(
        "--results_filename",
        type=str,
        default="results/results.csv",
        help="Output filename for the experiment results",
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
    parser.add_argument(
        "--keyphrase",
        type=str,
        default="acknowledge mistakes",
        help="The keyphrase to use for searching for features to vary during the experiment.",
    )
    args = parser.parse_args()

    load_dotenv()
    variant = goodfire.Variant(args.variant)
    if not os.path.exists(args.dataset_filename):
        print("Creating a new dataset for", args.dataset_filename)
        print("Loading MMLU dataset...")
        sample_df = load_and_format_mmlu_dataset(args.dataset_size)
        print("Gathering reasoning and answers from Llama/GPT...")
        dataset = create_dataset(sample_df, variant)
        print("Here is your dataset:")
        print(dataset)
        print("Removing incorrect answers...")
        dataset = remove_incorrect_answers(dataset)
        print("Saving dataset...")
        save_dataset(dataset, args.dataset_filename)
        print("Dataset saved to", args.dataset_filename)
    print("Using dataset from", args.dataset_filename, "for the experiment.")
    reasoning_dataset = pd.read_csv(args.dataset_filename)
    print("Searching features...")
    features = search_features(args.keyphrase, variant)
    print("Running experiment...")
    run_experiment(features, reasoning_dataset, variant, args.results_filename)
    print("Experiment results saved to", args.results_filename)


if __name__ == "__main__":
    main()
