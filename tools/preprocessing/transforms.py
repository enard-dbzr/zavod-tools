import abc
import warnings
from typing import Callable, Union

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


class Transform(abc.ABC):
    """
    Абстрактный базовый класс для преобразований данных.
    Все наследники должны реализовать метод __call__.
    """

    @abc.abstractmethod
    def __call__(self, x: pd.DataFrame, y: pd.DataFrame, borders: list[pd.DatetimeIndex]) -> tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Основной метод преобразования данных.
        
        Args:
            x: DataFrame с признаками
            y: DataFrame с целевыми переменными
            
        Returns:
            Кортеж из преобразованных (x, y)
            :param borders:
        """
        pass


class TransformPipeline(Transform):
    """
    Pipeline для последовательного применения преобразований.
    
    Attributes:
        transforms: Последовательность преобразований
    """

    def __init__(self, *transforms: Transform):
        self.transforms = transforms

    def __call__(self, x, y, borders):
        for t in self.transforms:
            x, y = t(x, y)

        return x, y


class SpecifiedColumnsTransform(Transform):
    """
    Применяет преобразование только к указанным столбцам.
    
    Attributes:
        transform: Преобразование
        columns: Список столбцов для обработки
    """

    def __init__(self, transform: Transform, columns: list[str]):
        self.transform = transform
        self.columns = columns

    def __call__(self, x, y, borders):
        """
        Выделяет указанные столбцы из x и y, применяет преобразование
        и объединяет результат обратно в исходные DataFrames.
        :param borders:
        """

        x_columns = [c for c in x.columns if c in self.columns]
        y_columns = [c for c in y.columns if c in self.columns]

        updated_x, updated_y = self.transform(x[x_columns], y[y_columns])

        x.loc[updated_x.index, updated_x.columns] = updated_x
        y.loc[updated_y.index, updated_y.columns] = updated_y
        return x, y


class FunctionTransform(Transform):
    """
    Преобразование-обёртка для произвольной функции.
    
    Attributes:
        fun: Функция для преобразования (x, y) -> (new_x, new_y)
    """

    def __init__(self, fun: Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]):
        self.fun = fun

    def __call__(self, x, y, borders):
        return self.fun(x, y)


class UnitedFunctionTransform(Transform):
    """
    Преобразование, работающее с объединёнными x и y.
    
    Attributes:
        fun: Функция для преобразования объединённого DataFrame (оказывается функция не должна менять порядок или количество столбцов, а то все сломается)
    """

    def __init__(self, fun: Callable[[pd.DataFrame], pd.DataFrame]):
        self.fun = fun

    def __call__(self, x, y, borders):
        df = pd.concat([x, y], axis=1)

        df = self.fun(df)

        return df.iloc[:, :len(x.columns)], df.iloc[:, len(x.columns):]


class BordersDependentTransform(Transform):
    def __init__(self, transform: Transform):
        self.transform = transform

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame, borders: list[pd.DatetimeIndex]) -> tuple[
        pd.DataFrame, pd.DataFrame]:

        borders = borders.copy() + [x.index.max()]
        for i in range(len(borders) - 1):
            # FIXME: самое последнее окно не учитывается
            x_split = x.loc[borders[i]:borders[i + 1]].iloc[:-1]
            y_split = y.loc[borders[i]:borders[i + 1]].iloc[:-1]

            x_split[:], y_split[:] = self.transform(x_split, y_split, [])

        return x, y


class AddTimeDelta(Transform):
    """
    Добавляет дельту - счётчик пропущенных значений для каждого интервала.
    """

    def __call__(self, x, y, borders):
        x, y = x.copy(), y.copy()

        for col in x.columns:
            df = x[col]
            # Группируем по последовательностям не-NaN значений
            x[f"{col}_delta"] = df.isna().groupby(df.notna().cumsum()).cumsum()

        for col in y.columns:
            df = y[col]
            y[f"{col}_delta"] = df.isna().groupby(df.notna().cumsum()).cumsum()

        return x, y


class Filler(Transform):
    """
    Универсальный филлер для заполнения пропущенных значений.
    
    Attributes:
        strategy (str/dict): Стратегия заполнения ('mean', 'median', 'mode', 
                            'ffill', 'bfill', 'interpolate'). Можно указать отдельно для колонок.
        method: Метод интерполяции ('linear', 'time', 'nearest' и т.д.)
        limit: Максимальное количество последовательных заполнений
        group_by: Колонка для группового заполнения
        use_global_stats: Использовать глобальные статистики (медиана/среднее и т.д.)
        custom_func: Пользовательская функция заполнения
    """

    def __init__(self,
                 strategy: Union[str, dict] = 'ffill',
                 constant_value: any = None,
                 method: str = 'linear',
                 limit: int = None,
                 group_by: str = None,
                 use_global_stats: bool = False,
                 custom_func: Callable[[pd.Series], pd.Series] = None):
        self.strategy = strategy
        self.constant_value = constant_value
        self.method = method
        self.limit = limit
        self.group_by = group_by
        self.use_global_stats = use_global_stats
        self.custom_func = custom_func

        self.global_stats = {}
        self._validate_parameters()

    def _validate_parameters(self):
        valid_strategies = ['mean', 'median', 'mode',
                            'ffill', 'bfill', 'interpolate']

        if isinstance(self.strategy, dict):
            for col, strat in self.strategy.items():
                if strat not in valid_strategies + ['custom']:
                    raise ValueError(f"Недопустимая стратегия для колонки {col}")
        elif self.strategy not in valid_strategies:
            raise ValueError(f"Недопустимая стратегия: {self.strategy}")

    def _get_strategy_for_column(self, col: str) -> str:
        """Возвращает стратегию для конкретной колонки"""
        if isinstance(self.strategy, dict):
            return self.strategy.get(col, 'ffill')  # default
        return self.strategy

    def _calc_global_stats(self, x: pd.DataFrame):
        """Вычисляет глобальные статистики для числовых колонок"""
        numeric_cols = x.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            strategy = self._get_strategy_for_column(col)

            if strategy == 'mean':
                self.global_stats[col] = x[col].mean()
            elif strategy == 'median':
                self.global_stats[col] = x[col].median()
            elif strategy == 'mode':
                self.global_stats[col] = x[col].mode()[0]

    def _fill_column(self,
                     series: pd.Series,
                     strategy: str) -> pd.Series:
        """Заполняет пропуски в одной колонке"""
        if self.custom_func:
            return self.custom_func(series)

        if strategy == 'mean':
            fill_value = self.global_stats[series.name] if self.use_global_stats else series.mean()
            return series.fillna(fill_value, limit=self.limit)

        if strategy == 'median':
            fill_value = self.global_stats[series.name] if self.use_global_stats else series.median()
            return series.fillna(fill_value, limit=self.limit)

        if strategy == 'mode':
            fill_value = self.global_stats[series.name] if self.use_global_stats else series.mode()[0]
            return series.fillna(fill_value, limit=self.limit)

        if strategy == 'ffill':
            return series.ffill(limit=self.limit)

        if strategy == 'bfill':
            return series.bfill(limit=self.limit)

        if strategy == 'interpolate':
            return series.interpolate(method=self.method, limit=self.limit)

        return series

    def _apply_filling(self, df):
        if self.use_global_stats and not self.global_stats:
            self._calc_global_stats(df)

        if self.group_by and self.group_by in df.columns:
            df = df.groupby(self.group_by).apply(
                lambda g: g.transform(
                    lambda col: self._fill_column(col, self._get_strategy_for_column(col.name))
                ))
        else:
            for col in df.columns:
                strategy = self._get_strategy_for_column(col)
                df[col] = self._fill_column(df[col], strategy)

        if self.use_global_stats and not self.global_stats:
            self._calc_global_stats(df)

        return df

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame, borders) -> tuple[pd.DataFrame, pd.DataFrame]:
        x, y = x.copy(), y.copy()

        x = self._apply_filling(x)
        y = self._apply_filling(y)

        return x, y


class Resample(Transform):
    """
    Ресемплирование временного ряда с заданной агрегацией по заданному правилу.
    
    Attributes:
        rule: Правило ресемплирования (например, '10min')
        aggregation: Функция агрегации данных (mean, max, min, median)
    """

    def __init__(self, rule="10min", aggregation='mean'):
        self.rule = rule
        self.aggregation = aggregation

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame, borders) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.concat([x, y], axis=1)
        df = getattr(df.resample(self.rule), self.aggregation)()
        return df.iloc[:, :len(x.columns)], df.iloc[:, len(x.columns):]


class KMeansClusterTransform(Transform):
    def __init__(self,
                 n_clusters: int = 8,
                 include_targets=False,
                 batch_size: int = 30_000,
                 max_buffer_size: int = None,
                 random_state: int = None,
                 cluster_column: str = 'cluster'):
        """
        Parameters:
        n_clusters: количество кластеров
        batch_size: размер батча для частичного обучения
        max_buffer_size: максимальный размер буфера для накопления данных
        """
        self.n_clusters = n_clusters
        self.include_targets = include_targets
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size
        self.random_state = random_state
        self.cluster_column = cluster_column

        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=batch_size
        )
        self.buffer = pd.DataFrame()
        self.is_initialized = False

    def _add_to_buffer(self, data: pd.DataFrame):
        """Накопление данных"""
        self.buffer = pd.concat([self.buffer, data], axis=0)
        if self.max_buffer_size is not None and len(self.buffer) > self.max_buffer_size:
            self.buffer = self.buffer.iloc[-self.max_buffer_size:]

    def _partial_fit(self):
        """Частичное обучение модели"""
        if not self.buffer.empty:
            data = self.buffer.to_numpy()

            self.model.partial_fit(data)
            self.is_initialized = True

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame, borders) -> tuple[pd.DataFrame, pd.DataFrame]:
        x_new = x.copy()

        self._add_to_buffer(x_new)

        if len(self.buffer) >= self.batch_size:
            self._partial_fit()

        # Предсказание кластеров если модель инициализирована
        if self.is_initialized:
            clusters = self.model.predict(x_new.to_numpy())
            x_new[self.cluster_column] = clusters
            unique_clusters = len(np.unique(clusters))
            if unique_clusters < self.n_clusters:
                warnings.warn(
                    f"\nОбнаружено только {unique_clusters} кластеров из {self.n_clusters}!\n",
                    RuntimeWarning
                )
        else:
            pass

        return x_new, y

    def force_fit(self, data: Union[pd.DataFrame, "PEMSDataset"] = None):
        """Принудительное обучение"""
        from tools.dataset import PEMSDataset
        if data is not None:
            if isinstance(data, PEMSDataset):
                data = data.x
            self._add_to_buffer(data)

        self._partial_fit()

        return self


class OneHotEncoderTransform(Transform):
    """
    Создает one-hot кодирование для указанной колонки.
    
    Attributes:
        column: Название колонки
    """

    def __init__(self, column: str, drop_original: bool = True, drop_first: bool = True):
        self.column = column
        self.drop_original = drop_original
        self.drop_first = drop_first

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame, borders) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.column not in x.columns:
            return x, y

        dummies = pd.get_dummies(x[self.column], prefix=self.column, drop_first=self.drop_first)
        x = pd.concat([x, dummies], axis=1)

        if self.drop_original:
            x = x.drop(columns=[self.column])

        return x, y
