from typing import Callable
import warnings
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

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame, borders) -> tuple[pd.DataFrame, pd.DataFrame]:
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


class IQRFilter(Transform):
    """
    Фильтрует выбросы по межквартильному размаху для указанных колонок.
    
    Attributes:
        columns: Список колонок для анализа (None = все числовые колонки)
        iqr_multiplier: Множитель для определения границ (по умолчанию 1.5)
    """
    def __init__(self, columns: list = None, iqr_multiplier: float = 1.5):
        self.columns = columns
        self.iqr_multiplier = iqr_multiplier

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame, borders) -> tuple[pd.DataFrame, pd.DataFrame]:
        x_filtered = x.copy()
        
        if self.columns is None:
            self.columns = x.select_dtypes(include=np.number).columns.tolist()
            
        for col in self.columns:
            if col in x.columns:
                q1 = x[col].quantile(0.25)
                q3 = x[col].quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - self.iqr_multiplier * iqr
                upper_bound = q3 + self.iqr_multiplier * iqr
                
                mask = (x[col] >= lower_bound) & (x[col] <= upper_bound)
                x_filtered = x_filtered[mask]
                y = y[mask]
                
        return x_filtered, y


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
                 use_latest: bool = False):
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
        missing_indices = [idx for idx in indices if idx not in x.index]
        
        if missing_indices and self.strict_mode and not self.use_latest:
            raise ValueError(f"Индексы не найдены: {missing_indices}")
        elif missing_indices and not self.use_latest:
            warnings.warn(f"Пропущены отсутствующие индексы: {missing_indices}", UserWarning)

        valid_indices = [idx for idx in indices if idx in x.index]

        # Грязный и медленный хак, чтоб срезы по датам получались...
        values = pd.concat([x.loc[idx, self.column_name] for idx in valid_indices])

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

        filtered_x = x[mask].copy()
        filtered_y = y[mask].copy()

        removed = len(x) - len(filtered_x)
        remaining = filtered_x[self.column_name].nunique()
        print(f"Удалено строк: {removed}")
        print(f"Уникальных значений осталось: {remaining}")

        return filtered_x, filtered_y

    def reset_accumulated(self):
        """Сброс накопленных данных"""
        self.accumulated_values = set()
