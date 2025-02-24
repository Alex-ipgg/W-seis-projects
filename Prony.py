import math
import bisect
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError
from joblib import Parallel, delayed
from scipy import signal, stats, linalg
from numpy.linalg import inv, pinv, eig
from scipy.linalg import toeplitz, hankel
from sklearn.metrics import mean_squared_error


class Prony:
    def __init__(self, traces):
        # Общие параметры
        self._sample_rate = 2  # шаг дискретизации
        self._method = 0  # 0 - LSM, 1 - MP
        self._num_terms = (
            0  # Количество синусоид для разложения в окне (0 - автоподбор)
        )
        self._win_size = 0  # Ширина окна в дискретах (0 - автоподбор)
        self._singular_number = 0  # Сингулярное число для MP
        self._task = 1  # 0 - фильтрация, 1 - трасса атрибутов
        self._shift = 1  # Сдвиг окна (если 0, то сдвиг на ширину окна)

        # Параметры фильтрации
        self._filt_freq = 20  # Доминантная частота для фильтрации
        self._spread = 10  # Ширина фильтрации (от self.spread - self.num_terms до self.spread + self.num_terms)
        self._limit = 1.5

        # Параметры атрибутов
        self._attribute = 0  # Трасса из: 0 - затухания, 1 - частоты, 2 - амплитуды, 3 - фазы, 4 - добротности

        # Дальше не менять
        self.traces = np.array(traces, dtype=np.float64)
        self.nsamp = len(self.traces[0])
        self.num_traces = len(self.traces)
        self.coef = [1000, 1000, 2, 1]

        self._update_dependent_variables()

    def _update_dependent_variables(self):
        """
        Описание : Обновляет зависимые переменные на основе текущих значений параметров.
        Действие :
        Пересчитывает такие параметры, как сдвиг окна (_shift), половину ширины окна (half_window), максимальный индекс для обработки (max_index) и перекрытие окон (_overlay).
        """

        self._method_str = (
            "prony_decomposition_least_squares"
            if self._method == 0
            else "prony_decomposition_matrix_pencil"
        )
        self._shift = self._win_size if self._shift == 0 else self._shift
        self.half_window = self._win_size // 2
        self.max_index = self.nsamp - self._win_size + 1
        self._overlay = self._win_size - self._shift

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        if value > 0:
            self._sample_rate = value
            self._update_dependent_variables()
        else:
            raise ValueError("Sample rate must be greater than 0")

    @property
    def method(self):
        return self._method_str

    @method.setter
    def method(self, value):
        if isinstance(value, int) and value in [0, 1]:
            self._method = value
        elif isinstance(value, str) and value in [
            "prony_decomposition_least_squares",
            "prony_decomposition_matrix_pencil",
        ]:
            self._method = 0 if value == "prony_decomposition_least_squares" else 1
        else:
            raise ValueError("Method error")
        self._update_dependent_variables()

    @property
    def filt_freq(self):
        return self._filt_freq

    @filt_freq.setter
    def filt_freq(self, value):
        if value > 0:
            self._filt_freq = value
            self._update_dependent_variables()
        else:
            raise ValueError("Filter frequency must be greater than 0")

    @property
    def spread(self):
        return self._spread

    @spread.setter
    def spread(self, value):
        if value >= 0:
            self._spread = value
            self._update_dependent_variables()
        else:
            raise ValueError("Spread must be non-negative")

    @property
    def num_terms(self):
        return self._num_terms

    @num_terms.setter
    def num_terms(self, value):
        if value >= 0:
            self._num_terms = value
            self._update_dependent_variables()
        else:
            raise ValueError("Number of terms must be non-negative")

    @property
    def win_size(self):
        return self._win_size

    @win_size.setter
    def win_size(self, value):
        if value >= 0:
            self._win_size = value
            self._update_dependent_variables()
        else:
            raise ValueError("Window size must be non-negative")

    @property
    def overlay(self):
        return self._overlay

    @overlay.setter
    def overlay(self, value):
        if value >= 0:
            self._overlay = value
            self._update_dependent_variables()
        else:
            raise ValueError("Overlay must be non-negative")

    @property
    def limit(self):
        return self._limit

    @limit.setter
    def limit(self, value):
        if value > 0:
            self._limit = value
            self._update_dependent_variables()
        else:
            raise ValueError("Limit must be greater than 0")

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        if value in [0, 1]:
            self._task = value
            self._update_dependent_variables()
        else:
            raise ValueError("Task must be 0 or 1")

    @property
    def attribute(self):
        return self._attribute

    @attribute.setter
    def attribute(self, value):
        if value in range(4):
            self._attribute = value
            self._update_dependent_variables()
        else:
            raise ValueError("Attribute must be between 0 and 4")

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, value):
        if value >= 0:
            self._shift = value
            self._update_dependent_variables()
        else:
            raise ValueError("Shift must be positive")

    @property
    def singular_number(self):
        return self._singular_number

    @singular_number.setter
    def singular_number(self, value):
        if value >= 0:
            self._singular_number = value
            self._update_dependent_variables()
        else:
            raise ValueError("Singular number must be non-negative")

    def prony_decomposition_least_squares(self, data):
        """
        Описание : Выполняет разложение сигнала методом наименьших квадратов.
        Параметры :
        data: Входной сигнал.
        Возвращает :
        Коэффициенты затухания, частоты, амплитуды и фазы.
        """

        data = data.tolist()
        nsamp = len(data)

        toeplitz_matrix = np.asarray(
            toeplitz(
                data[self.num_terms - 1 : nsamp - 1 : 1],
                data[self.num_terms - 1 : 0 : -1] + [data[0]],
            )
        )
        a = -1 * np.dot(
            np.dot(
                inv(np.dot(toeplitz_matrix.transpose(), toeplitz_matrix)),
                toeplitz_matrix.transpose(),
            ),
            np.asarray(data[self.num_terms : -1 : 1] + [data[(-1)]]),
        )
        coef_toep = np.concatenate((np.array([1]), a))
        roots = np.roots(coef_toep)

        damping_factor = np.log(np.abs(roots)) / self.sample_rate

        frequency = np.arctan2(roots.imag, roots.real) / (2 * np.pi * self.sample_rate)

        vandermonde_matrix = np.vander(roots, increasing=True).transpose()

        coef_vander = np.dot(
            np.dot(
                inv(np.dot(vandermonde_matrix.transpose(), vandermonde_matrix)),
                vandermonde_matrix.transpose(),
            ),
            np.asarray(data[0 : self.num_terms]),
        )

        amplitude = np.abs(coef_vander)

        phase = np.arctan2(coef_vander.imag, coef_vander.real)

        return damping_factor, frequency, amplitude, phase

    def prony_decomposition_matrix_pencil(self, data):
        """
        Описание : Выполняет разложение сигнала методом матричного пучка.
        Параметры :
        data: Входной сигнал.
        Возвращает :
        Коэффициенты затухания, частоты, амплитуды и фазы.
        """

        data = data.tolist()
        nsamp = len(data)

        hankel_matrix = np.asarray(
            hankel(
                data[0 : nsamp - self.num_terms : 1],
                data[nsamp - self.num_terms - 1 : nsamp : 1],
            )
        )  # First column and last row of the matrix
        # print(hankel_matrix.shape)
        nlines, ncols = hankel_matrix.shape

        hankel_matrix_1 = hankel_matrix[:, 0 : ncols - 1]

        hankel_matrix_2 = hankel_matrix[:, 1:ncols]

        eigenvalues, right_eigenvectors = eig(
            np.dot(pinv(hankel_matrix_1), hankel_matrix_2)
        )

        damping_factor = np.log(np.abs(eigenvalues)) / self.sample_rate

        frequency = np.arctan2(eigenvalues.imag, eigenvalues.real) / (
            2 * np.pi * self.sample_rate
        )

        vandermonde_matrix = np.vander(eigenvalues, increasing=True).transpose()

        coef_vander = np.dot(
            self._fil_Singular_matrix(data=vandermonde_matrix),
            np.asarray(data[0 : self.num_terms : 1]),
        )

        amplitude = np.abs(coef_vander)

        phase = np.arctan2(coef_vander.imag, coef_vander.real)

        return damping_factor, frequency, amplitude, phase

    def _fil_Singular_matrix(self, data):
        """
        Описание : Вычисляет псевдообратную матрицу с использованием сингулярного разложения.
        Параметры :
        data: Матрица данных.
        Возвращает :
        Псевдообратную матрицу.
        """

        U, S, Vh = np.linalg.svd(data)

        # Создаем булевый массив-маску для больших сингулярных значений
        mask = S > self.singular_number

        # Инвертируем только большие сингулярные значения
        S_inv = np.zeros_like(S)
        S_inv[mask] = 1 / S[mask]

        # Вычисляем псевдообратную матрицу, используя инвертированные сингулярные значения
        # и соответствующие сингулярные векторы
        data_pinv = Vh.T @ (S_inv[:, np.newaxis] * U.T)

        # При необходимости берем комплексное сопряжение
        return data_pinv.conjugate()

    def prony_approximation(
        self, num_terms, damping_factor, frequency, amplitude, phase
    ):
        """
        Описание : Восстанавливает сигнал на основе параметров Прони.
        Параметры :
        num_terms: Количество компонентов.
        damping_factor: Коэффициенты затухания.
        frequency: Частоты.
        amplitude: Амплитуды.
        phase: Фазы.
        Возвращает :
        Восстановленный сигнал.
        """

        # Оптимизированные векторные вычисления
        exponents = np.arange(num_terms)[:, None]
        z = np.exp((damping_factor + 2j * np.pi * frequency) * self.sample_rate)
        return (z**exponents).dot(amplitude * np.exp(1j * phase))

    def _rms_error_calc(self, data, estimated_data):
        """
        Описание : Вычисляет среднеквадратичную ошибку между исходным и аппроксимированным сигналами.
        Параметры :
        data: Исходный сигнал.
        estimated_data: Аппроксимированный сигнал.
        Возвращает :
        Среднеквадратичную ошибку.
        """

        rms_error = np.sqrt(mean_squared_error(data, estimated_data))
        return np.round(rms_error, 4)

    def _find_optimal_num_terms(self, data):
        """
        Описание : Автоматически подбирает оптимальное количество членов разложения (num_terms) для минимизации ошибки.
        Параметры :
        data: Входной сигнал.
        Возвращает :
        Минимальную ошибку и оптимальное количество членов.
        """

        warnings.filterwarnings("ignore")

        min_err = np.inf
        optimal_num = None

        for i in tqdm(range(2, math.floor(self.win_size * 0.8)), desc="num_terms calc"):
            self.num_terms = i

            try:
                S = self.reconstruct_signal(data)
                if S is None or len(S) < 3 or np.isnan(S).any():
                    continue

                r = self._rms_error_calc(data[1:-2], S[1:-2])
                if not np.isfinite(r):
                    continue

            except Exception:
                continue

            if r < min_err:
                min_err = r
                optimal_num = i

        return min_err, optimal_num

    def reconstruct_signal(self, data):
        """
        Описание : Выполняет разложение сигнала с использованием скользящего окна.
        Параметры :
        data: Входной сигнал.
        Возвращает :
        Аппроксимированный сигнал.
        """

        step = self._win_size - self._overlay
        c_trace = np.zeros(len(data), dtype=np.float64)

        for a in range(0, len(data) - self._win_size + 1, step):
            window = data[a : a + self._win_size]
            damping, frequency, amplitude, phase = getattr(self, self.method)(window)
            S = self.prony_approximation(
                self._win_size, damping, frequency, amplitude, phase
            )

            if self._overlay == 0:
                c_trace[a : a + self._win_size] = S
            else:
                overlap_region = a + self._overlay
                c_trace[a:overlap_region] = (
                    c_trace[a:overlap_region] + S[: self._overlay]
                ) / 2
                c_trace[overlap_region : a + self._win_size] = S[self._overlay :]

        return c_trace

    def filter_prony_components(self, data, damping, freq, amp, phase):
        """
        Описание : Фильтрует компоненты Прони на основе заданных критериев частоты.
        Параметры :
        data: Исходный сигнал.
        damping: Коэффициенты затухания.
        freq: Частоты.
        amp: Амплитуды.
        phase: Фазы.
        Возвращает :
        Отфильтрованный сигнал.
        """

        # Векторизованная маска для фильтрации
        mask = (np.abs(freq) * 1000 >= self.filt_freq - self.spread) & (
            np.abs(freq) * 1000 <= self.filt_freq + self.spread
        )

        return self.prony_approximation(
            len(data), damping[mask], freq[mask], amp[mask], phase[mask]
        ).real

    def apply_prony_filter(self, data):
        """
        Описание: Выполняет фильтрацию сигнала в окне методом Прони.
        Параметры:
        data: Входной сигнал.
        Возвращает:
        Отфильтрованный сигнал.
        """

        c_trace = np.zeros(len(data), dtype="float64")
        method_func = getattr(self, self.method)

        if self.win_size == 0:
            damping_factor, frequency, amplitude, phase = method_func(data)
            c_trace[:] = self.filter_prony_components(
                data, damping_factor, frequency, amplitude, phase
            )
        else:
            a, i = 0, 0
            step = self.win_size - self.overlay

            while a + self.win_size <= len(data):
                n_trace = data[a : a + self.win_size]

                damping_factor, frequency, amplitude, phase = method_func(n_trace)
                S = self.filter_prony_components(
                    n_trace, damping_factor, frequency, amplitude, phase
                )

                if i == 0 or self.overlay == 0:
                    c_trace[a : a + self.win_size] = S
                else:
                    overlay_end = a + self.overlay
                    if self.overlay != 0:
                        c_trace[a:overlay_end] = (
                            c_trace[a:overlay_end] + S[: self.overlay]
                        ) / 2
                    c_trace[overlay_end : a + self.win_size] = S[self.overlay :]

                a += step
                i += 1

        return c_trace.real

    def trace_of_attributes(self, data):
        """
        Описание : Вычисляет атрибуты (затухание, частоту, амплитуду, фазу) для одной трассы.
        Параметры :
        trace: Одна трасса данных.
        Возвращает :
        Четыре массива: затухание, частота, амплитуда, фаза.
        """

        # Создаем массивы для хранения результатов
        c_traces = np.zeros(
            (4, len(data)), dtype=np.float64
        )  # Используем float32 для лучшей производительности

        # Предвычисляем константы
        half_window = self.half_window
        max_index = len(data) - self.win_size + 1
        norm_factor = 1 / self.num_terms

        # Векторизованное вычисление
        for a in range(max_index):
            # Извлекаем подмассив данных
            n_trace = data[a : a + self.win_size]

            # Разлагаем данные с помощью decomposition_method
            decomposed = self.decomposition_method(n_trace)

            # Вычисляем индекс для записи результатов
            index = a + half_window

            if index < len(data):
                # Записываем нормализованные значения в соответствующие массивы
                c_traces[: len(decomposed), index] = [
                    np.linalg.norm(component) * norm_factor for component in decomposed
                ]

        # Возвращаем результаты как кортеж
        return tuple(c_traces[i] * self.coef[i] for i in range(4))

    def trace_of_attributes_shifted(self, data):
        """
        Описание: Вычисляет атрибуты трассы (затухание, частоту, амплитуду, фазу)
        с учетом сдвига окна.
        Параметры:
        data: Входной сигнал.
        Возвращает:
        Кортеж из четырех массивов, содержащих затухание, частоту, амплитуду и фазу.
        """

        # Вычисляем длину c_trace
        c_trace_length = len(data)

        # Создаем четыре массива для хранения результатов
        c_traces = [np.zeros(c_trace_length, dtype="float64") for _ in range(4)]

        # Проходим по данным с шагом shift
        for a in range(0, self.max_index, self.shift):
            # Извлекаем подмассив данных
            n_trace = data[a : a + self.win_size]

            # Разлагаем данные с помощью decomposition_method
            decomposed = self.decomposition_method(n_trace)

            # Вычисляем индекс для записи результатов
            index = a + self.half_window

            if index < c_trace_length:
                # Записываем нормализованные значения в соответствующие массивы
                for i, component in enumerate(decomposed):
                    c_traces[i][index] = np.linalg.norm(component) / self.num_terms

                # Заполняем промежуточные точки с использованием линейной интерполяции
                if a + self.shift < self.max_index:
                    next_index = a + self.shift + self.half_window
                    if next_index < c_trace_length:
                        for i in range(4):
                            # Вычисляем количество промежуточных точек
                            num_intermediate_points = next_index - index - 1

                            if num_intermediate_points > 0:
                                # Линейная интерполяция
                                interpolated_values = np.linspace(
                                    c_traces[i][index],
                                    c_traces[i][next_index],
                                    num_intermediate_points + 2,
                                )[
                                    1:-1
                                ]  # Исключаем начальную и конечную точки

                                # Записываем интерполированные значения
                                c_traces[i][
                                    index + 1 : next_index
                                ] = interpolated_values

        # Возвращаем результаты как кортеж из четырех массивов
        return tuple(c_traces[i] * self.coef[i] for i in range(4))

    def plot_section(self):
        """
        Описание: Основной метод для фильрации и получения трасс атрибутов.
        Автоматически определяет параметры, если они не заданы, и обрабатывает все ненулевые трассы.
        Возвращает:
        Результат обработки в виде матрицы отфильтрованных сигналов или набора матриц атрибутов.
        """

        # Определение постоянных трасс
        first_elements = self.traces[:, 0]
        constant_mask = np.all(self.traces == first_elements[:, None], axis=1)
        self.non_constant_indices = np.where(~constant_mask)[0]
        if not self.non_constant_indices.size:
            ind = self.num_traces // 2
        else:
            ind = self.non_constant_indices[0]

        # Автоподбор параметров, если они не заданы
        if self._win_size == 0:
            self.win_size = math.ceil(
                ((1 / self.filt_freq) * 3) / (self.sample_rate / 1000)
            )
            print(f"win = {self.win_size} for freq = {self.filt_freq}")

        if self.num_terms == 0:
            n = self._find_optimal_num_terms(self.traces[ind])
            self.num_terms = self.win_size // 2 if n[1] in [np.inf, None] else n[1]
        else:
            S = self.reconstruct_signal(self.traces[ind])
            r = self._rms_error_calc(self.traces[ind][1:-2], S[1:-2])
            print(f"Error = {r}")

        print(f"num_terms = {self.num_terms}")

        # Обработка непостоянных трасс
        def process_trace(idx):
            """
            Описание: Обрабатывает отдельную трассу в зависимости от текущей задачи (фильтрация или вычисление атрибутов).
            Параметры:
            idx: Индекс обрабатываемой трассы.
            Возвращает:
            Результат обработки трассы (отфильтрованный сигнал или набор атрибутов).
            """

            try:
                trace = self.traces[idx]
                if self.task == 0:
                    result = self.apply_prony_filter(trace)
                elif self.task == 1:
                    func = (
                        self.trace_of_attributes
                        if self.shift == 1
                        else self.trace_of_attributes_shifted
                    )
                    result = func(trace)
                return result
            except Exception as e:
                warnings.warn(f"Error in trace {idx}: {e}")
                return (
                    np.zeros_like(trace)
                    if self.task == 0
                    else [np.zeros_like(trace) for _ in range(4)]
                )

        # Параллельная обработка трасс
        results = Parallel(n_jobs=-1)(
            delayed(process_trace)(idx)
            for idx in tqdm(self.non_constant_indices, desc="Processing traces")
        )

        # Формирование выходных данных
        if self.task == 0:
            if self.non_constant_indices.size == 0:
                return np.zeros_like(self.traces, dtype=np.float64)
            else:
                # Определение размера на основе первого результата
                sample_length = len(results[0])
                num_traces = self.traces.shape[0]
                m = np.zeros((sample_length, num_traces), dtype=np.float64)
                for idx, result in zip(self.non_constant_indices, results):
                    m[:, idx] = result
                return m

        elif self.task == 1:
            if self.non_constant_indices.size == 0:
                return [np.zeros_like(self.traces) for _ in range(4)]
            else:
                # Определение размера для атрибутов
                sample_length = len(results[0][0])
                num_traces = self.traces.shape[0]
                m1, m2, m3, m4 = [
                    np.zeros((sample_length, num_traces), dtype=np.float16)
                    for _ in range(4)
                ]
                for idx, result in zip(self.non_constant_indices, results):
                    m1[:, idx], m2[:, idx], m3[:, idx], m4[:, idx] = result
                return m1, m2, m3, m4
