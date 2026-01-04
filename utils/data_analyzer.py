import pandas as pd
import re
import os
from pathlib import Path
from typing import Dict, List, Optional


class DataAnalyzer:
    """A small utility focused on loading data and providing basic accessors.

    Added: a normalization API for matching preferences (female/male).
    - get_matching_preferences(): normalizes column names and boolean values,
      and returns [female_id, male_id, want_female, want_male].
    """

    def __init__(self, data_dir: Optional[str] = None, matching_file: Optional[str] = None):
        """Load data and perform basic categorization.

        Parameters:
        -----------
        data_dir : str, optional
            Directory containing conversation CSV files.
        matching_file : str, optional
            Path to the matching data CSV file.
        """
        self.data_dir = data_dir
        self.matching_df = None
        self.conversation_data = {}

        if matching_file and os.path.exists(matching_file):
            self.matching_df = pd.read_csv(matching_file)

        if data_dir and os.path.exists(data_dir):
            self._load_conversations_from_dir(data_dir)
            self._classify_conversations()
        else:
            self.first_round_files = []
            self.second_round_files = []

    def _load_conversations_from_dir(self, data_dir: str):
        """Load all CSV files from the specified directory."""
        data_path = Path(data_dir)
        for csv_file in data_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                self.conversation_data[csv_file.name] = df
            except Exception as e:
                print(f"Warning: Failed to load {csv_file.name}: {e}")

    def _classify_conversations(self):
        """Classify conversation files into first/second round buckets."""
        self.first_round_files = []
        self.second_round_files = []

        for filename in self.conversation_data.keys():
            # Example: extract the round number ("1") from "01_1_1_A_F088_M073_concat.csv"
            match = re.search(r'^\d+_(\d+)_', filename)
            if match:
                round_num = match.group(1)
                if round_num == '1':
                    self.first_round_files.append(filename)
                elif round_num == '2':
                    self.second_round_files.append(filename)

    def get_matching_data(self) -> Optional[pd.DataFrame]:
        """Return the matching DataFrame (if loaded)."""
        return self.matching_df

    def _normalize_bool(self, s: pd.Series) -> pd.Series:
        """Normalize a series into boolean values."""
        # Accept True/False, TRUE/FALSE, 1/0, yes/no
        return s.apply(lambda x: True if str(x).strip().lower() in {"true", "1", "t", "y", "yes"}
                                 else (False if str(x).strip().lower() in {"false", "0", "f", "n", "no"}
                                       else pd.NA))

    def get_matching_preferences(self) -> pd.DataFrame:
        """
        Return normalized matching preferences (female/male).

        Returns
        -------
        DataFrame with columns: female_id(Int64), male_id(Int64), want_female(boolean), want_male(boolean)
        """
        if self.matching_df is None:
            raise ValueError("Matching data has not been loaded.")

        df = self.matching_df.copy()

        # Normalize column names (Japanese -> English)
        col_map = {
            "女性ID": "female_id",
            "男性ID": "male_id",
            "女性の交換意欲": "want_female",
            "男性の交換意欲": "want_male",
        }

        for jp, en in list(col_map.items()):
            if en in df.columns:
                continue
            if jp in df.columns:
                df = df.rename(columns={jp: en})

        # Required columns check
        required = {"female_id", "male_id", "want_female", "want_male"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"Matching preference columns not found (missing: {missing}). "
                "Please verify the columns provided by your data loader."
            )

        df["female_id"] = pd.to_numeric(df["female_id"], errors="coerce").astype("Int64")
        df["male_id"] = pd.to_numeric(df["male_id"], errors="coerce").astype("Int64")
        df["want_female"] = self._normalize_bool(df["want_female"]).astype("boolean")
        df["want_male"] = self._normalize_bool(df["want_male"]).astype("boolean")

        return df[["female_id", "male_id", "want_female", "want_male"]]

    def get_conversations(self, round_number: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Get conversation data.

        Returns:
        --------
        A dict of conversation data: {filename: dataframe}
        """
        if round_number == 1:
            return {filename: self.conversation_data[filename]
                   for filename in self.first_round_files}
        elif round_number == 2:
            return {filename: self.conversation_data[filename]
                   for filename in self.second_round_files}
        elif round_number is None:
            return self.conversation_data
        else:
            raise ValueError("round_number must be 1, 2, or None.")

    def get_conversation_files(self, round_number: Optional[int] = None) -> List[str]:
        """
        Get the list of conversation file names.

        Parameters:
        -----------
        round_number: 1 (first round), 2 (second round), None (all)

        Returns:
        --------
        List of file names.
        """
        if round_number == 1:
            return self.first_round_files.copy()
        elif round_number == 2:
            return self.second_round_files.copy()
        elif round_number is None:
            return list(self.conversation_data.keys())
        else:
            raise ValueError("round_number must be 1, 2, or None.")

    def get_single_conversation(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Get conversation data for a specific file.

        Parameters:
        -----------
        filename: file name or file path

        Returns:
        --------
        pandas.DataFrame or None
        """
        # If it's just a file name
        if filename in self.conversation_data:
            return self.conversation_data[filename]

        # If it's a full path, load directly
        if os.path.exists(filename):
            try:
                return pd.read_csv(filename)
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                return None

        # Fall back to searching by basename
        base_name = os.path.basename(filename)
        if base_name in self.conversation_data:
            return self.conversation_data[base_name]

        return None
