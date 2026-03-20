import os
import pandas as pd
import pm4py


class LogsDataProcessor:
    def __init__(self, name, filepath, columns, dir_path="./data"):
        """Provides support for processing raw logs into prefixes.

        Args:
            name: str: Dataset name
            filepath: str: Path to raw logs dataset
            columns: list: Column names in the raw log
            dir_path: str: Path to directory for the dataset
        """
        self._name = name
        self._filepath = filepath
        self._org_columns = columns
        self._dir_path = dir_path

        if not os.path.exists(f"{dir_path}/{self._name}"):
            os.makedirs(f"{dir_path}/{self._name}")
        self._filepath = f"{dir_path}/{self._name}/{self._name}.csv"

    def load_df(self, sort_temporally=True, add_eoc=False):
        """Load and preprocess the raw event log.

        Args:
            sort_temporally: Whether to sort events by timestamp
            add_eoc: Whether to add [EOC] (End of Case)

        Returns:
            Preprocessed DataFrame
        """
        # Load CSV file if it exists, otherwise load XES file
        if os.path.exists(self._filepath):
            df = pd.read_csv(self._filepath)
        else:
            self._filepath = f"{self._dir_path}/{self._name}/{self._name}.xes"
            event_log = pm4py.read_xes(self._filepath)
            df = pm4py.convert_to_dataframe(event_log)
        df = df[self._org_columns]

        # Standardize column names
        df.columns = ["case:concept:name", "concept:name", "time:timestamp"]

        # Clean activity names and timestamps
        df["concept:name"] = df["concept:name"].astype(str)
        df["concept:name"] = df["concept:name"].str.lower()
        df["concept:name"] = df["concept:name"].str.replace(" ", "-")
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])

        # Sort by timestamp if requested
        if sort_temporally:
            df = df.sort_values(by=["case:concept:name", "time:timestamp"])

        # Add [EOC] if requested
        if add_eoc:
            last_events = df.groupby("case:concept:name").tail(1).copy()
            last_events["concept:name"] = "[EOC]"
            last_events["time:timestamp"] = last_events["time:timestamp"] + pd.Timedelta(seconds=1)
            df = pd.concat([df, last_events]).sort_values(by=["case:concept:name", "time:timestamp"])

        # Convert timestamps to string format for output
        df["time:timestamp"] = df["time:timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

        return df

    def create_prefixes(self, df, prefix_length=None):
        """Create prefixes from the event log.

        Args:
            df: DataFrame containing the event log
            prefix_length: Optional maximum length for prefixes (handled by LogsDataLoader)

        Returns:
            DataFrame with prefixes
        """
        # Group events by case
        grouped = df.groupby("case:concept:name")

        # Initialize results lists
        case_ids = []
        prefixes = []
        positions = []
        timestamps = []
        next_acts = []

        for case_id, case_df in grouped:
            # Skip cases with only one event
            if len(case_df) <= 1:
                continue

            activities = case_df["concept:name"].tolist()
            case_timestamps = case_df["time:timestamp"].tolist()

            # Create prefixes for each position in the case
            for i in range(len(activities) - 1):
                prefix = " ".join(activities[:i + 1])

                # Store the data
                case_ids.append(case_id)
                prefixes.append(prefix)
                positions.append(i)
                timestamps.append(case_timestamps[i])
                next_acts.append(activities[i + 1])

        # Create dataframe
        result_df = pd.DataFrame({
            "case_id": case_ids,
            "prefix": prefixes,
            "k": positions,
            "last_event_time": timestamps,
            "next_act": next_acts
        })

        return result_df