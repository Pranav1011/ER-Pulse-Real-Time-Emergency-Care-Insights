import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

class BaselineCalculator:
    """Calculate seasonal baselines for metrics (hour, day-of-week)."""

    def __init__(self, data_path: str = "data/processed/admissions.csv"):
        self.data_path = Path(data_path)
        self.baselines: Dict[str, pd.DataFrame] = {}
        self._load_and_calculate()

    def _load_and_calculate(self):
        """Load data and calculate baselines."""
        if not self.data_path.exists():
            # Use parent directory path for when running from backend/
            alt_path = Path("../data/processed/admissions.csv")
            if alt_path.exists():
                self.data_path = alt_path
            else:
                raise FileNotFoundError(f"Data not found: {self.data_path}")

        df = pd.read_csv(self.data_path)
        df['admittime'] = pd.to_datetime(df['admittime'])
        df['day_of_week'] = df['admittime'].dt.dayofweek

        # ED Wait Time baselines by (hour, day_of_week)
        wait_baselines = df.groupby(['hour', 'day_of_week'])['ed_wait_time'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        wait_baselines.columns = ['hour', 'day_of_week', 'mean', 'std', 'count']
        wait_baselines['std'] = wait_baselines['std'].fillna(wait_baselines['std'].mean())
        self.baselines['ed_wait_time'] = wait_baselines

        # Admission rate baselines (count per hour/day)
        admission_counts = df.groupby(['hour', 'day_of_week']).size().reset_index(name='count')
        admission_baselines = admission_counts.groupby(['hour', 'day_of_week'])['count'].agg([
            'mean', 'std'
        ]).reset_index()
        admission_baselines['std'] = admission_baselines['std'].fillna(1.0)
        self.baselines['admission_rate'] = admission_baselines

    def get_baseline(self, metric: str, hour: int, day_of_week: int) -> Tuple[float, float]:
        """Get mean and std for a metric at specific hour/day."""
        if metric not in self.baselines:
            return 0.0, 1.0

        df = self.baselines[metric]
        match = df[(df['hour'] == hour) & (df['day_of_week'] == day_of_week)]

        if match.empty:
            # Fallback to hour-only baseline
            match = df[df['hour'] == hour]
            if match.empty:
                return df['mean'].mean(), df['std'].mean()

        return float(match['mean'].iloc[0]), float(match['std'].iloc[0])

    def calculate_z_score(self, metric: str, value: float, hour: int, day_of_week: int) -> float:
        """Calculate z-score for a value against seasonal baseline."""
        mean, std = self.get_baseline(metric, hour, day_of_week)
        if std == 0:
            std = 1.0
        return (value - mean) / std

    def get_all_baselines(self) -> Dict:
        """Return all baselines as dict."""
        return {k: v.to_dict('records') for k, v in self.baselines.items()}
