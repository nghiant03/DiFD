"""Intel Lab dataset loader.

Loads and preprocesses the Intel Berkeley Research Lab sensor dataset.
Data format: date time epoch moteid temperature humidity light voltage
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from DiFD.datasets.base import BaseDataset


class IntelLabDataset(BaseDataset):
    """Loader for Intel Berkeley Research Lab sensor dataset.

    The dataset contains readings from 54 sensors deployed in the
    Intel Berkeley Research lab between February 28th and April 5th, 2004.

    Columns: date, time, epoch, moteid, temperature, humidity, light, voltage
    """

    COLUMNS = ["date", "time", "epoch", "moteid", "temp", "humid", "light", "volt"]

    def __init__(self, data_path: str | Path) -> None:
        """Initialize the Intel Lab dataset loader.

        Args:
            data_path: Path to data.txt file.
        """
        super().__init__(data_path)

    @property
    def name(self) -> str:
        return "intel_lab"

    @property
    def feature_columns(self) -> list[str]:
        return ["temp", "humid", "light", "volt"]

    @property
    def group_column(self) -> str:
        return "moteid"

    @property
    def timestamp_column(self) -> str:
        return "timestamp"

    def load(self) -> pd.DataFrame:
        """Load the Intel Lab dataset from disk.

        Returns:
            DataFrame with columns: timestamp, moteid, temp, humid, light, volt
        """
        df = pd.read_csv(
            self.data_path,
            sep=r"\s+",
            header=None,
            names=self.COLUMNS,
            na_values=[""],
            on_bad_lines="skip",
        )

        df["timestamp"] = pd.to_datetime(
            df["date"] + " " + df["time"],
            format="%Y-%m-%d %H:%M:%S.%f",
            errors="coerce",
        )

        df = df.drop(columns=["date", "time", "epoch"])
        logger.debug(f"Loaded {len(df)} rows from {self.data_path}")

        df = df.dropna(subset=["timestamp", "moteid"])
        logger.debug(f"Dropped rows with invalid timestamps or moteids. Remaining rows: {len(df)}")

        df["moteid"] = df["moteid"].astype(int)

        df = df[df["volt"] > 2.4]
        logger.debug(f"Dropped voltage readings <= 2.4V. Remaining rows: {len(df)}")

        df = df.sort_values(["moteid", "timestamp"]).reset_index(drop=True)

        return df

    def preprocess(
        self,
        df: pd.DataFrame,
        resample_freq: str = "30s",
        interpolation_method: str = "linear",
    ) -> pd.DataFrame:
        """Resample and interpolate the dataset per mote.

        Groups by moteid, resamples to fixed frequency, and interpolates
        missing values. Does NOT mix data between different motes.

        Args:
            df: Raw dataframe from load().
            resample_freq: Frequency string for resampling (e.g., "5min").
            interpolation_method: Method for interpolation (e.g., "linear").

        Returns:
            Preprocessed DataFrame with regular time intervals per mote.
        """
        feature_cols = self.feature_columns
        processed_dfs: list[pd.DataFrame] = []

        for mote_id, group in df.groupby("moteid"):
            group = group.set_index("timestamp")
            group = group[feature_cols]

            logger.debug(f"Processing moteid {mote_id} with {len(group)} records.")
            resampled = group.resample(resample_freq).mean()

            logger.debug(f"Remaining missing values after resampling: {resampled.isna().sum().sum()}")
            if interpolation_method == "ffill":
                interpolated = resampled.ffill().bfill()
            else:
                interpolated = resampled.interpolate(method=interpolation_method)
                interpolated = interpolated.ffill().bfill()

            assert not interpolated.isna().any().any(), f"Missing values remain after interpolation for moteid {mote_id}"

            interpolated["moteid"] = mote_id
            interpolated = interpolated.reset_index()

            processed_dfs.append(interpolated)

        if not processed_dfs:
            return pd.DataFrame(columns=["timestamp", "moteid", *feature_cols])

        result = pd.concat(processed_dfs, ignore_index=True)
        result = result.sort_values(["moteid", "timestamp"]).reset_index(drop=True)

        return result
