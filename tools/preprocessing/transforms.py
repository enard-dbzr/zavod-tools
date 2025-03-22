import abc
from typing import Callable
import warnings 
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

class Transform(abc.ABC):
    """
    Абстрактный базовый класс для преобразований данных.
    Все наследники должны реализовать метод __call__.
    """
    @abc.abstractmethod
    def __call__(self, x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Основной метод преобразования данных.
        
        Args:
            x: DataFrame с признаками
            y: DataFrame с целевыми переменными
            
        Returns:
            Кортеж из преобразованных (x, y)
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

    def __call__(self, x, y):
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

    def __call__(self, x, y):
        """
        Выделяет указанные столбцы из x и y, применяет преобразование
        и объединяет результат обратно в исходные DataFrames.
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

    def __call__(self, x, y):
        return self.fun(x, y)


class UnitedFunctionTransform(Transform):
    """
    Преобразование, работающее с объединёнными x и y.
    
    Attributes:
        fun: Функция для преобразования объединённого DataFrame (оказывается функция не должна менять порядок или количество столбцов, а то все сломается)
    """
    def __init__(self, fun: Callable[[pd.DataFrame], pd.DataFrame]):
        self.fun = fun

    def __call__(self, x, y):
        df = pd.concat([x, y], axis=1)

        df = self.fun(df)

        return df.iloc[:, :len(x.columns)], df.iloc[:, len(x.columns):]


class FfillWithTimeDelta(Transform):
    """
    Заполнение пропусков методом forward fill с добавлением дельты - счётчика 
    пропущенных значений для каждого интервала.
    """
    def __call__(self, x, y):
        x, y = x.copy(), y.copy()
        
        for col in x.columns:
            df = x[col]
            # Группируем по последовательностям не-NaN значений
            x[f"{col}_delta"] = df.isna().groupby(df.notna().cumsum()).cumsum()
        
        for col in y.columns:
            df = y[col]
            y[f"{col}_delta"] = df.isna().groupby(df.notna().cumsum()).cumsum()

        # Заполнение пропусков
        x.ffill(inplace=True)
        y.ffill(inplace=True)
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
        self.aggregation=aggregation

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.concat([x, y], axis=1)
        df = getattr(df.resample(self.rule), self.aggregation)()
        return df.iloc[:, :len(x.columns)], df.iloc[:, len(x.columns):]

class KMeansClusterTransformer(Transform):
    def __init__(self, 
                 n_clusters: int = 8,
                 include_targets=False,
                 batch_size: int = 1000,
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
        self.include_targets=include_targets
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

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    def force_fit(self):
        """Принудительное обучение на остатках данных в буфере"""
        self._partial_fit()