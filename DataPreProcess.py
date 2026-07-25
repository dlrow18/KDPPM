import os
import argparse
import sys

# Add the project root to the Python path
sys.path.append(
    os.path.abspath(os.path.dirname(__file__))
)

from PreProcessing.DataProcesser import LogsDataProcessor


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocess raw event logs for next-activity prediction."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, e.g. BPIC15_2"
    )

    parser.add_argument(
        "--dir_path",
        type=str,
        default="./data",
        help="Root directory containing the dataset folders"
    )

    parser.add_argument(
        "--raw_log_file",
        type=str,
        default=None,
        help=(
            "Path to the raw CSV event log. "
            "If omitted, the path is inferred as "
            "<dir_path>/<dataset>/<dataset>.csv"
        )
    )

    parser.add_argument(
        "--sort_temporally",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort cases chronologically"
    )

    parser.add_argument(
        "--add_eoc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add an [EOC] activity to each case"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    raw_log_file = args.raw_log_file

    if raw_log_file is None:
        raw_log_file = os.path.join(
            args.dir_path,
            args.dataset,
            f"{args.dataset}.csv"
        )

    if not os.path.isfile(raw_log_file):
        raise FileNotFoundError(
            f"Raw event log not found: {raw_log_file}"
        )

    print(
        f"[Preprocessing] Started | "
        f"dataset={args.dataset} | "
        f"input={raw_log_file}"
    )

    data_processor = LogsDataProcessor(
        name=args.dataset,
        filepath=raw_log_file,
        columns=[
            "case:concept:name",
            "concept:name",
            "time:timestamp",
        ],
        dir_path=args.dir_path,
    )

    event_log = data_processor.load_df(
        sort_temporally=args.sort_temporally,
        add_eoc=args.add_eoc,
    )

    prefixes_df = data_processor.create_prefixes(event_log)

    output_dir = os.path.join(
        args.dir_path,
        args.dataset,
        "processed",
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "prefixes.csv",
    )
    prefixes_df.to_csv(output_path, index=False)

    print(
        f"[Preprocessing] Completed | "
        f"prefixes={len(prefixes_df)} | "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()