import abc
from typing import Callable

import pandas as pd


class Transform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass


class TransformPipeline(Transform):
    def __init__(self, *transforms: Transform):
        self.transforms = transforms

    def __call__(self, x, y):
        for t in self.transforms:
            x, y = t(x, y)

        return x, y


class SpecifiedColumnsTransform(Transform):
    def __init__(self, transform: Transform, columns: list[str]):
        self.transform = transform
        self.columns = columns

    def __call__(self, x, y):
        x_columns = [c for c in x.columns if c in self.columns]
        y_columns = [c for c in y.columns if c in self.columns]

        updated_x, updated_y = self.transform(x[x_columns], y[y_columns])

        x.loc[updated_x.index, updated_x.columns] = updated_x
        y.loc[updated_y.index, updated_y.columns] = updated_y
        return x, y


class FunctionTransform(Transform):
    def __init__(self, fun: Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]):
        self.fun = fun

    def __call__(self, x, y):
        return self.fun(x, y)


class UnitedFunctionTransform(Transform):
    def __init__(self, fun: Callable[[pd.DataFrame], pd.DataFrame]):
        self.fun = fun

    def __call__(self, x, y):
        df = pd.concat([x, y], axis=1)

        df = self.fun(df)

        return df.iloc[:, :len(x.columns)], df.iloc[:, len(x.columns):]


class FfillWithTimeDelta(Transform):
    def __call__(self, x, y):
        x, y = x.copy(), y.copy()

        for col in x.columns:
            df = x[col]
            x[f"{col}_delta"] = df.isna().groupby(df.notna().cumsum()).cumsum()

        for col in y.columns:
            df = y[col]
            y[f"{col}_delta"] = df.isna().groupby(df.notna().cumsum()).cumsum()

        x.ffill(inplace=True)
        y.ffill(inplace=True)
        return x, y


class Resample(Transform):
    def __init__(self, rule="10min"):
        self.rule = rule

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.concat([x, y], axis=1)

        df = df.resample(self.rule).mean()

        return df.iloc[:, :len(x.columns)], df.iloc[:, len(x.columns):]
