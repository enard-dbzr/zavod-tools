import numpy as np
import pandas as pd

from tools.preprocessing.transforms import Transform


class FilterOutliers(Transform):
    def __init__(self, file_paths=None, limits=None):
        if limits is None:
            limits = {}
        if file_paths is None:
            file_paths = []

        self.limits = limits
        self.file_paths = file_paths

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.concat([x, y], axis=1)

        for path in self.file_paths:
            outliers_df = pd.read_excel(path)

            for idx, (col, begin, end, method, *_) in outliers_df.iterrows():
                if pd.notnull(col):
                    if method == "nan" or pd.isna(method):
                        df.loc[begin:end, col] = np.nan

        # comp = df.copy()
        for col, (lb, ub) in self.limits.items():
            df[col] = df[col].clip(lb, ub)

        # print((comp != df)[comp.notna()].sum())

        return df.iloc[:, :len(x.columns)], df.iloc[:, len(x.columns):]
