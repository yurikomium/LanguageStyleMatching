import pandas as pd
import re
import os
from pathlib import Path
from typing import Dict, List, Optional


class DataAnalyzer:
    """データ読み込みと基本アクセスに特化したクラス

    追加: 交換意欲（女性/男性）の正規化APIを提供
    - get_matching_preferences(): 列名正規化とブール化を行い、
      [female_id, male_id, want_female, want_male] を返す
    """

    def __init__(self, data_dir: Optional[str] = None, matching_file: Optional[str] = None):
        """データ読み込みと基本分類

        Parameters:
        -----------
        data_dir : str, optional
            対話データが格納されているディレクトリ
        matching_file : str, optional
            マッチングデータのCSVファイルパス
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
        """指定ディレクトリからCSVファイルを読み込む"""
        data_path = Path(data_dir)
        for csv_file in data_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                self.conversation_data[csv_file.name] = df
            except Exception as e:
                print(f"Warning: Failed to load {csv_file.name}: {e}")

    def _classify_conversations(self):
        """1回目・2回目の対話データを分類"""
        self.first_round_files = []
        self.second_round_files = []

        for filename in self.conversation_data.keys():
            # 例: "01_1_1_A_F088_M073_concat.csv" から会話回数「1」を抽出
            match = re.search(r'^\d+_(\d+)_', filename)
            if match:
                round_num = match.group(1)
                if round_num == '1':
                    self.first_round_files.append(filename)
                elif round_num == '2':
                    self.second_round_files.append(filename)

    def get_matching_data(self) -> Optional[pd.DataFrame]:
        """マッチングデータを取得"""
        return self.matching_df

    def _normalize_bool(self, s: pd.Series) -> pd.Series:
        """ブール値を正規化"""
        # True/False, TRUE/FALSE, 1/0, yes/no を許容
        return s.apply(lambda x: True if str(x).strip().lower() in {"true", "1", "t", "y", "yes"}
                                 else (False if str(x).strip().lower() in {"false", "0", "f", "n", "no"}
                                       else pd.NA))

    def get_matching_preferences(self) -> pd.DataFrame:
        """
        交換意欲（女性/男性）を正規化して返す。

        Returns
        -------
        DataFrame with columns: female_id(Int64), male_id(Int64), want_female(boolean), want_male(boolean)
        """
        if self.matching_df is None:
            raise ValueError("マッチングデータが読み込まれていません")

        df = self.matching_df.copy()

        # 列名正規化（日本語→英語）
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

        # 必須列チェック
        required = {"female_id", "male_id", "want_female", "want_male"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"交換意欲列が見つかりません（不足: {missing}）。data_loader で供給される列をご確認ください。")

        df["female_id"] = pd.to_numeric(df["female_id"], errors="coerce").astype("Int64")
        df["male_id"] = pd.to_numeric(df["male_id"], errors="coerce").astype("Int64")
        df["want_female"] = self._normalize_bool(df["want_female"]).astype("boolean")
        df["want_male"] = self._normalize_bool(df["want_male"]).astype("boolean")

        return df[["female_id", "male_id", "want_female", "want_male"]]

    def get_conversations(self, round_number: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        対話データを取得

        Returns:
        --------
        辞書形式の対話データ {filename: dataframe}
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
            raise ValueError("round_number は 1, 2, または None を指定してください")

    def get_conversation_files(self, round_number: Optional[int] = None) -> List[str]:
        """
        ファイル名のリストを取得

        Parameters:
        -----------
        round_number: 1 (1回目), 2 (2回目), None (全て)

        Returns:
        --------
        ファイル名のリスト
        """
        if round_number == 1:
            return self.first_round_files.copy()
        elif round_number == 2:
            return self.second_round_files.copy()
        elif round_number is None:
            return list(self.conversation_data.keys())
        else:
            raise ValueError("round_number は 1, 2, または None を指定してください")

    def get_single_conversation(self, filename: str) -> Optional[pd.DataFrame]:
        """
        特定のファイルの対話データを取得

        Parameters:
        -----------
        filename: ファイル名またはファイルパス

        Returns:
        --------
        pandas.DataFrame or None
        """
        # ファイル名だけの場合
        if filename in self.conversation_data:
            return self.conversation_data[filename]

        # フルパスの場合は直接読み込み
        if os.path.exists(filename):
            try:
                return pd.read_csv(filename)
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                return None

        # ベース名で検索
        base_name = os.path.basename(filename)
        if base_name in self.conversation_data:
            return self.conversation_data[base_name]

        return None
