from typing import Callable, Optional
import warnings
import numpy as np
import pandas as pd

from tools.preprocessing.transforms import Transform


class RangeFileFilter(Transform):
    """
    Фильтрует значения по указанному файлу exel.
    """

    def __init__(self, file_paths=None, limits=None):
        """
        :param file_paths: Пути до файлов exel структуры.
        :param limits: Пределы измерений величин для метода фильтрации lim.
        """

        if limits is None:
            limits = {}
        if file_paths is None:
            file_paths = []

        self.limits = limits
        self.file_paths = file_paths

    def __call__(self, parts, borders):
        df, *merge_meta = self._merge_parts(parts)

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

        return self._split_parts(df, *merge_meta)


class IQRFilter(Transform):
    """
    Фильтрует выбросы по интерквантильному размаху для указанных колонок.
    """
    def __init__(self, iqr_multiplier: Optional[float] = 1.5, width=0.5, gap="1min",
                 column_specified_settings: dict[str, tuple[float, float, str]] = None):
        """
        :param iqr_multiplier: Множитель размаха для определения границ.
        :param width: Ширина интерквантильного размаха.
        :param gap: Ширина окрестности для зачистки.
        :param column_specified_settings: Словарь настроек для колонок.
        """
        self.iqr_multiplier = iqr_multiplier
        self.width = width
        self.gap = gap
        self.settings = column_specified_settings or {}

        self.test = None

    def __call__(self, parts, borders):
        df, *merge_meta = self._merge_parts(parts)

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
                
        return self._split_parts(df, *merge_meta)


class ColumnValueFilter(Transform):
    """
    Универсальный фильтр данныx.

    :ivar column_name: Колонка для фильтрации.
    :ivar exclude_values: Значения для исключения.
    :ivar include_values: Значения для сохранения.
    :ivar exclude_indices: Индексы для извлечения исключаемых значений.
    :ivar include_indices: Индексы для извлечения сохраняемых значений.
    :ivar mode: Режим работы ('exclude', 'include', 'auto').
    :ivar min_count: Минимальное количество для авторежима.
    :ivar condition: Кастомное условие фильтрации.
    :ivar strict_mode: Вызывать ошибку при отсутствии индексов.
    :ivar use_latest: Накапливать историю значений из колонки.
    :ivar gap: Временная окрестность для исключения.
    :ivar shift: Сдвиг маски.
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

    def __call__(self, parts, borders):
        df, *merge_meta = self._merge_parts(parts)

        if self.column_name not in df.columns:
            raise ValueError(f"Колонка {self.column_name} не найдена")

        if self.condition:
            mask = self.condition(df[self.column_name])
            return self._split_parts(df[mask], *merge_meta)

        filter_values = self._prepare_filter_values(df)

        if self.mode == 'include' or self.include_indices:
            mask = df[self.column_name].isin(filter_values)
        else:
            mask = ~df[self.column_name].isin(filter_values)

        mask = mask.rolling(self.gap, center=True).min() == 1
        mask = mask.shift(freq=self.shift).reindex(df.index, fill_value=True)

        filtered_df = df[mask].copy()

        removed = len(df) - len(filtered_df)
        remaining = filtered_df[self.column_name].unique()
        print(f"Удалено строк: {removed}")
        print(f"Уникальных значений осталось: {remaining}")

        return self._split_parts(filtered_df, *merge_meta)

    def reset_accumulated(self):
        """Сброс накопленных данных"""
        self.accumulated_values = set()
