import os
import argparse
import sys

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from PreProcessing.DataProcesser import LogsDataProcessor

# Parse arguments
parser = argparse.ArgumentParser(description="Process raw event logs for CIL2D.")

parser.add_argument("--dataset",
                    type=str,
                    default="Sepsis",
                    help="dataset name")

parser.add_argument("--dir_path",
                    type=str,
                    default="./data",
                    help="path to store processed data")

parser.add_argument("--raw_log_file",
                    type=str,
                    default="./data/Sepsis/Sepsis.csv",
                    help="path to raw csv log file")

parser.add_argument("--sort_temporally",
                    type=bool,
                    default=True,
                    help="sort cases by timestamp")

parser.add_argument("--add_eoc",
                    type=bool,
                    default=False,
                    help="add [EOC] to each case")

args = parser.parse_args()

if __name__ == "__main__":
    # Create processor for the dataset
    data_processor = LogsDataProcessor(
        name=args.dataset,
        filepath=args.raw_log_file,
        columns=["case:concept:name", "concept:name", "time:timestamp"],
        dir_path=args.dir_path
    )

    # Load and preprocess the event log
    event_log = data_processor.load_df(
        sort_temporally=args.sort_temporally,
        add_eoc=args.add_eoc
    )

    # Create prefixes
    prefixes_df = data_processor.create_prefixes(event_log)

    # print(f"Created {len(prefixes_df)} prefixes for {args.dataset}")

    # Save the preprocessed data to a file for later use
    output_dir = f"{args.dir_path}/{args.dataset}/processed"
    os.makedirs(output_dir, exist_ok=True)
    prefixes_df.to_csv(f"{output_dir}/prefixes.csv", index=False)
    # print(f"Prefixes saved to {output_dir}/prefixes.csv")