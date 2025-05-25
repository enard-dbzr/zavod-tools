import abc
import warnings
from copy import deepcopy
from typing import Callable, Union

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


class Transform(abc.ABC):
    """
    Абстрактный базовый класс для преобразований данных.
    Все наследники должны реализовать метод :meth:`Transform.__call__`.
    """

    @abc.abstractmethod
    def __call__(self, parts: dict[str, pd.DataFrame], borders: list[pd.DatetimeIndex]) -> dict[str, pd.DataFrame]:
        """
        Основной метод преобразования данных.
        
        :arg parts: Наборы табличных данных
        :arg borders: Границы независсимых кусков
            
        :return: Преобразовнные наборы табличных данных
        """
        pass

    @classmethod
    def _merge_parts(cls, parts: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
        """
        Объединяет словарь DataFrame'ов в один DataFrame.
        """
        keys = list(parts)
        col_sizes = {k: df.shape[1] for k, df in parts.items()}
        merged = pd.concat([parts[k] for k in keys], axis=1)
        return merged, keys, col_sizes

    @classmethod
    def _split_parts(cls, df: pd.DataFrame, keys: list[str], col_sizes: dict[str, int]) -> dict[str, pd.DataFrame]:
        """
        Делит один DataFrame обратно на части по ключам и размерам колонок.
        """
        result = {}
        start = 0
        for k in keys:
            end = start + col_sizes[k]
            result[k] = df.iloc[:, start:end]
            start = end
        return result


class TransformPipeline(Transform):
    """
    Pipeline для последовательного применения преобразований.
    
    :ivar transforms: Последовательность трансформаций
    """

    def __init__(self, *transforms: Transform):
        self.transforms = transforms

    def __call__(self, parts, borders):
        for t in self.transforms:
            parts = t(parts, borders)

        return parts


class SpecifiedColumnsTransform(Transform):
    """
    Применяет преобразование только к указанным столбцам.
    НЕ РАБОТАЕТ, если во вложенной трансформации удалять строки, столбцы или части.

    :ivar transform: Преобразование
    :ivar columns: Список столбцов для обработки
    """

    def __init__(self, transform: Transform, columns: list[str]):
        self.transform = transform
        self.columns = columns

    def __call__(self, parts, borders):
        """
        Выделяет указанные столбцы из x и y, применяет преобразование
        и объединяет результат обратно в исходные DataFrames.
        """

        selected_parts = {
            key: df[df.columns.intersection(self.columns)]
            for key, df in parts.items()
        }

        updated = self.transform(selected_parts, borders)

        for key, updated_df in updated.items():
            if key not in parts:
                parts[key] = updated_df
                continue

            parts[key].loc[updated_df.index, updated_df.columns] = updated_df

        return parts


class FunctionTransform(Transform):
    """
    Преобразование-обёртка для произвольной функции.
    """

    def __init__(self, fun: Callable[[dict[str, pd.DataFrame]], dict[str, pd.DataFrame]]):
        """
        :arg fun: функция, принимающая именованные части и возвращающая преобразованные.
        """
        self.fun = fun

    def __call__(self, parts, borders):
        return self.fun(parts)


class UnitedFunctionTransform(Transform):
    """
    Преобразование, работающее с объединёнными частями.
    """

    def __init__(self, fun: Callable[[pd.DataFrame], pd.DataFrame]):
        """
        :arg fun: Функция для преобразования объединённого DataFrame
        (оказывается функция не должна менять порядок или количество столбцов, а то все сломается).
        """
        self.fun = fun

    def __call__(self, parts, borders):
        df, keys, col_sizes = self._merge_parts(parts)
        df = self.fun(df)
        return self._split_parts(df, keys, col_sizes)


class BordersDependentTransform(Transform):
    """
    Применяет преобразование независимо на разных кусках.
    """

    def __init__(self, transform: Transform):
        self.transform = transform

    def __call__(self, parts, borders: list[pd.DatetimeIndex]):
        result_parts = {key: [] for key in parts}

        for i in range(len(borders) - 1):
            eps = parts["x"].index[0].resolution

            split = {
                key: df.loc[borders[i]: borders[i + 1] - eps] for key, df in parts.items()
            }

            transformed = self.transform(split, [])

            for key in transformed:
                result_parts[key].append(transformed[key])

        return {key: pd.concat(dfs) for key, dfs in result_parts.items()}


class AddTimeDelta(Transform):
    """
    Добавляет дельту — счётчик пропущенных значений для каждого столбца в каждой части.
    """
    def __init__(self, part_label: str = None):
        """
        :arg part_label: Определяет в какую часть будет сохраняться информация. По умолчанию добавляется в ту же часть.
        """

        self.part_label = part_label

    def __call__(self, parts: dict[str, pd.DataFrame], borders: list[pd.DatetimeIndex]) -> dict[str, pd.DataFrame]:
        updated_parts = {}

        if self.part_label is not None:
            updated_parts[self.part_label] = parts["x"][[]].copy()

        for key, df in parts.items():
            df = df.copy()
            for col in df.columns:
                is_na = df[col].isna()
                groups = df[col].notna().cumsum()

                delta = is_na.groupby(groups).cumsum()
                if self.part_label is None:
                    df[f"{col}_delta"] = delta
                else:
                    updated_parts[self.part_label][f"{col}_delta"] = delta

            updated_parts[key] = df

        return updated_parts


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

    def __call__(self, parts, borders):
        transformed = {key: self._apply_filling(df.copy()) for key, df in parts.items()}

        return transformed


class Resample(Transform):
    """
    Ресемплирование временного ряда с заданной агрегацией по заданному правилу.
    """

    def __init__(self, rule="10min", aggregation='mean'):
        """
        :arg rule: Правило ресемплирования (например, '10min')
        :arg aggregation: Функция агрегации данных (mean, max, min, median)
        """
        self.rule = rule
        self.aggregation = aggregation

    def __call__(self, parts, borders):
        df, *merge_meta = self._merge_parts(parts)
        df = getattr(df.resample(self.rule), self.aggregation)()
        return self._split_parts(df, *merge_meta)


class KMeansClusterTransform(Transform):
    """
    Кластеризатор точек.
    """

    def __init__(self,
                 n_clusters: int = 8,
                 batch_size: int = 30_000,
                 max_buffer_size: int = None,
                 random_state: int = None,
                 cluster_column: str = 'cluster',
                 interpolator: Transform = None):
        """
        :arg n_clusters: количество кластеров
        :arg batch_size: размер батча для частичного обучения
        :arg max_buffer_size: максимальный размер буфера для накопления данных
        :arg interpolator: Трансформ для заполнения нанов. По умолчанию: TransformPipeline(Filler("ffill"), Filler("bfill")).
        """
        from tools.preprocessing.scalers import NormalScaler

        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size
        self.random_state = random_state
        self.cluster_column = cluster_column
        self.interpolator = interpolator or TransformPipeline(
            Filler("ffill"),
            Filler("bfill"),
            NormalScaler()
        )

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

    def __call__(self, parts, borders):
        interpolated_df, *_ = self._merge_parts(self.interpolator(parts, borders))

        if interpolated_df.isna().values.any():
            raise ValueError("В данных не должно содержаться NaN. Проверь interpolator.")

        self._add_to_buffer(interpolated_df)

        if len(self.buffer) >= self.batch_size:
            self._partial_fit()

        # Предсказание кластеров если модель инициализирована
        if self.is_initialized:
            clusters = self.model.predict(interpolated_df.to_numpy())
            parts["x"].loc[:, self.cluster_column] = clusters
            unique_clusters = len(np.unique(clusters))
            if unique_clusters < self.n_clusters:
                warnings.warn(
                    f"\nОбнаружено только {unique_clusters} кластеров из {self.n_clusters}!\n",
                    RuntimeWarning
                )
        else:
            pass

        return parts

    def force_fit(self, data: Union[pd.DataFrame, "PEMSDataset"] = None):
        """Принудительное обучение"""
        from tools.dataset import PEMSDataset
        if data is not None:
            if isinstance(data, PEMSDataset):
                data = data.dfs["x"]  # FIXME: брать не только иксы
            self._add_to_buffer(data)

        self._partial_fit()

        return self


class OneHotEncoderTransform(Transform):
    """
    Создает one-hot кодирование для указанной колонки. Работает только по X.
    """

    def __init__(self, column: str, drop_original: bool = True, drop_first: bool = True):
        """
        :param column: Название колонки.
        :param drop_original: Удалять ли колонку из которой были получены значения.
        :param drop_first: Удалять ли первый класс.
        """
        self.column = column
        self.drop_original = drop_original
        self.drop_first = drop_first

    def __call__(self, parts, borders):
        # TODO: Реализовать поддержку по любой части, а не только по x.

        if self.column not in parts["x"]:
            return parts

        dummies = pd.get_dummies(parts["x"][self.column], prefix=self.column, drop_first=self.drop_first)
        parts["x"] = pd.concat([parts["x"], dummies], axis=1)

        if self.drop_original:
            parts["x"] = parts["x"].drop(columns=[self.column])

        return parts


class MakeSnapshotTransform(Transform):
    """Сохраняет копию текущих колонок в новую часть."""

    def __init__(self, part_label: str, prefix: str = "snapshot_"):
        """
        :param part_label: Название новой части.
        :param prefix: Префикс колонок в новой части.
        """

        self.part_label = part_label
        self.prefix = prefix

    def __call__(self, parts, borders):
        df, *_ = self._merge_parts(parts)
        parts[self.part_label] = df.copy().add_prefix(self.prefix)

        return parts
