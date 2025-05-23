from typing import Callable, Optional
import warnings
import numpy as np
import pandas as pd

from tools.preprocessing.transforms import Transform


class RangeFileFilter(Transform):
    def __init__(self, file_paths=None, limits=None):
        if limits is None:
            limits = {}
        if file_paths is None:
            file_paths = []

        self.limits = limits
        self.file_paths = file_paths

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame, borders) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.concat([x, y], axis=1)

        for path in self.file_paths:
            outliers_df = pd.read_excel(path)

            for idx, (col, begin, end, method, *_) in outliers_df.iterrows():
                if pd.notnull(col):
                    if method == "nan" or pd.isna(method):
                        df.loc[begin:end, col if col != "*" else slice(None)] = np.nan

        # comp = df.copy()
        for col, (lb, ub) in self.limits.items():
            df[col] = df[col].clip(lb, ub)

        # print((comp != df)[comp.notna()].sum())

        return df.iloc[:, :len(x.columns)], df.iloc[:, len(x.columns):]


class IQRFilter(Transform):
    """
    Фильтрует выбросы по межквартильному размаху для указанных колонок.
    
    Attributes:
        columns: Список колонок для анализа (None = все числовые колонки)
        iqr_multiplier: Множитель для определения границ (по умолчанию 1.5)
        column_specified_settings: Словарь настроек для колонок
    """
    def __init__(self, iqr_multiplier: Optional[float] = 1.5, width=0.5, gap="1min",
                 column_specified_settings: dict[str, tuple[float, float, str]] = None):
        self.iqr_multiplier = iqr_multiplier
        self.width = width
        self.gap = gap
        self.settings = column_specified_settings or {}

        self.test = None

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame, borders) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.concat([x, y], axis=1)

        # self.test = df.copy()
            
        for col in df.columns:
            multiplier, width, gap = self.settings.get(col, (self.iqr_multiplier, self.width, self.gap))

            if multiplier is None or width is None:
                warnings.warn(f"Skipped column {col}")
                continue

            q1 = df[col].quantile(0.5 - width / 2)
            q3 = df[col].quantile(0.5 + width / 2)
            iqr = q3 - q1

            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            mask = mask.rolling(gap, center=True).max().astype(bool)
            df.loc[mask, col] = np.nan

            print(f"Removed {mask.sum()} ({mask.sum() / df[col].count() * 100 :.1f}%) values from {col} ")
                
        return df.iloc[:, :len(x.columns)], df.iloc[:, len(x.columns):]


class ColumnValueFilter(Transform):
    """
    Универсальный фильтр данныx.

    Attributes:
        column_name (str): Колонка для фильтрации
        exclude_values (list): Значения для исключения
        include_values (list): Значения для сохранения
        exclude_indices (list): Индексы для извлечения исключаемых значений
        include_indices (list): Индексы для извлечения сохраняемых значений
        mode (str): Режим работы ('exclude', 'include', 'auto')
        min_count (int): Минимальное количество для авторежима
        condition (callable): Кастомное условие фильтрации
        strict_mode (bool): Вызывать ошибку при отсутствии индексов
        use_latest (bool): Накапливать историю значений из колонки
        gap (str): Временная окрестность для исключения
        shift (str): Сдвиг маски
    """
    def __init__(self,
                 column_name: str,
                 exclude_values: list = None,
                 include_values: list = None,
                 exclude_indices: list = None,
                 include_indices: list = None,
                 mode: str = 'exclude',
                 min_count: int = 1000,
                 condition: Callable[[pd.Series], bool] = None,
                 strict_mode: bool = True,
                 use_latest: bool = False,
                 gap: str = "1min",
                 shift: str = "0min"):
        self.column_name = column_name
        self.exclude_values = exclude_values or []
        self.include_values = include_values or []
        self.exclude_indices = exclude_indices or []
        self.include_indices = include_indices or []
        self.mode = mode
        self.min_count = min_count
        self.condition = condition
        self.strict_mode = strict_mode
        self.use_latest = use_latest
        self.gap = gap
        self.shift = shift
        self.accumulated_values = set()

        self._validate_parameters()

    def _validate_parameters(self):
        valid_modes = ['exclude', 'include', 'auto']
        if self.mode not in valid_modes:
            raise ValueError(f"Недопустимый режим. Допустимые: {valid_modes}")
            
        if self.exclude_indices and self.include_indices:
            raise ValueError("Используйте либо exclude_indices, либо include_indices")

    def _get_values_from_indices(self, x: pd.DataFrame, indices: list) -> list:
        """Извлекает значения из колонки по указанным индексам"""
        splits = []
        for idx in indices:
            l, r = idx if isinstance(idx, tuple) else (idx, idx)
            s = x.loc[l:r, self.column_name]

            if s.empty and not self.use_latest:
                warnings.warn(f"Пропущен отсутствующий индекс: {idx}", UserWarning)
                if self.strict_mode:
                    raise ValueError(f"Индекс не найден: {idx}")

            splits.append(x.loc[l:r, self.column_name])

        values = pd.concat(splits)

        return values.unique().tolist()

    def _prepare_filter_values(self, x: pd.DataFrame) -> list:
        """Формирует итоговый список значений для фильтрации"""
        values = []

        if self.mode == 'exclude':
            values.extend(self.exclude_values)
        elif self.mode == 'include':
            values.extend(self.include_values)

        if self.exclude_indices:
            values.extend(self._get_values_from_indices(x, self.exclude_indices))
        if self.include_indices:
            values.extend(self._get_values_from_indices(x, self.include_indices))

        if self.mode == 'auto':
            if self.use_latest and not self.value_counts.empty:
                values.extend(self.value_counts[self.value_counts < self.min_count].index.tolist())
            else:
                value_counts = x[self.column_name].value_counts()
                values.extend(value_counts[value_counts < self.min_count].index.tolist())

        if self.use_latest:
            self.accumulated_values.update(values)
            values = self.accumulated_values
        # print(self.accumulated_values)
        
        return list(set(values))

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame, borders) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.column_name not in x.columns:
            raise ValueError(f"Колонка {self.column_name} не найдена")

        if self.condition:
            mask = self.condition(x[self.column_name])
            return x[mask], y[mask]

        filter_values = self._prepare_filter_values(x)

        if self.mode == 'include' or self.include_indices:
            mask = x[self.column_name].isin(filter_values)
        else:
            mask = ~x[self.column_name].isin(filter_values)

        mask = mask.rolling(self.gap, center=True).min() == 1
        mask = mask.shift(freq=self.shift).reindex(x.index, fill_value=True)

        filtered_x = x[mask].copy()
        filtered_y = y[mask].copy()

        removed = len(x) - len(filtered_x)
        remaining = filtered_x[self.column_name].unique()
        print(f"Удалено строк: {removed}")
        print(f"Уникальных значений осталось: {remaining}")

        return filtered_x, filtered_y

    def reset_accumulated(self):
        """Сброс накопленных данных"""
        self.accumulated_values = set()
