import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermitenorm


class Wavelet:
    def __init__(self, wavelet_name):
        self._wavelet_name = wavelet_name.lower()
        self._signal_length = 2000  # длина записи в мс
        self._sample_rate = 2  # шаг выборки в мс
        self._central_time = None  # центральное время импульса в мс
        self._dom_freq = 10  # центральная частота в Гц
        self._num_cycles = 6  # кол-во циклов

        self.file_path = "C:\\Users\\ALEKS\\copy\\coefficients.env"

        self._update_dependent_variables()

    def _update_dependent_variables(self):
        if self._central_time == None:
            self._central_time = self._signal_length // 2

        # Преобразование значений в секунды
        self._signal_length_sec = self._signal_length / 1000.0
        self._sample_rate_sec = self._sample_rate / 1000.0
        self._central_time_sec = self._central_time / 1000.0

        self._time = np.arange(
            0, self._signal_length_sec, self._sample_rate_sec
        )  # Массив времени
        self._time_shifted = (
            self._time - self._central_time_sec
        )  # Массив времени, центрированный вокруг t_0
        self._sigma = self._num_cycles / (
            2 * np.pi * self._dom_freq
        )  # Стандартное отклонение

    @property
    def signal_length(self):
        """Длина записи в мс."""
        return self._signal_length

    @signal_length.setter
    def signal_length(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Signal length must be a number.")
        if value <= 0:
            raise ValueError("Signal length must be positive.")
        if value <= 3:
            raise ValueError("Too low length of signal.")

        self._signal_length = value
        self._central_time = value // 2
        self._update_dependent_variables()

    @property
    def sample_rate(self):
        """Шаг выборки в мс."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Sample rate must be a number.")
        if value <= 0:
            raise ValueError("Sample rate must be positive.")
        if self.signal_length // value <= 3:
            raise ValueError("Too high value of sample rate.")

        self._sample_rate = value
        self._update_dependent_variables()

    @property
    def central_time(self):
        """Центральное время импульса в мс."""
        return self._central_time

    @central_time.setter
    def central_time(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Central time must be a number.")
        if value <= 0:
            raise ValueError("Central time must be positive.")
        if value > self.signal_length:
            raise ValueError("Too high value of central time.")

        self._central_time = value * 2
        self._update_dependent_variables()

    @property
    def wavelet_name(self):
        return self._wavelet_name

    @wavelet_name.setter
    def wavelet_name(self, value):
        self._wavelet_name = value.lower()
        self._update_dependent_variables()

    @property
    def dom_freq(self):
        """Центральная частота в Гц."""
        return self._dom_freq

    @dom_freq.setter
    def dom_freq(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Dominant frequency must be a number.")
        if value <= 0:
            raise ValueError("Dominant frequency must be positive.")

        self._dom_freq = value
        self._update_dependent_variables()

    @property
    def num_cycles(self):
        """Количество циклов."""
        return self._num_cycles

    @num_cycles.setter
    def num_cycles(self, value):
        if not isinstance(value, int):
            raise ValueError("Number of cycles must be a number (int).")
        if value <= 1:
            raise ValueError("Too low number of cycles.")
        if value > self.signal_length:
            raise ValueError("Too high value of central time.")

        self._num_cycles = value
        self._update_dependent_variables()

    @property
    def signal_length_sec(self):
        """Длина записи в секундах."""

        return self._signal_length_sec

    @property
    def sample_rate_sec(self):
        """Шаг выборки в секундах."""

        return self._sample_rate_sec

    @property
    def central_time_sec(self):
        """Центральное время импульса в секундах."""

        return self._central_time_sec

    @property
    def time(self):
        """Массив времени."""

        return self._time

    @property
    def time_shifted(self):
        """Массив времени, центрированный вокруг t_0."""

        return self._time_shifted

    @property
    def sigma(self):
        """Стандартное отклонение."""

        return self._sigma

    def generate_wavelet(self):
        """Генерирует вейвлет в соответствии с заданным названием."""

        if self._wavelet_name == "ricker":
            return self.ricker()

        elif self._wavelet_name.startswith("morlet"):
            return self.morlet()[int(self._wavelet_name[-1]) - 1]

        elif self._wavelet_name.startswith("gaus"):
            return self.gaus(int(self._wavelet_name[-1]))

        elif self._wavelet_name.startswith(("db", "coif", "sym")):
            # Извлекаем базовое название и число после него
            base_wavelet_name = ""
            number = ""
            for base in ("db", "coif", "sym"):
                if self._wavelet_name.startswith(base):
                    base_wavelet_name = base
                    number = self._wavelet_name[len(base) :]
                    break

            if base_wavelet_name and number.isdigit():
                return self.daubechies(base_wavelet_name, int(number))

        else:
            raise ValueError(f"Неизвестное название вейвлета: {self._wavelet_name}")

    def get_wavelet_list(self):
        """
        Генерирует список вейвлетов.

        Возвращает:
            wavelet_list (numpy.ndarray): Список с названиями доступных вейвлетов.
        """

        wavelet_list = (
            ["ricker"]
            + [f"morlet{n}" for n in range(1, 4)]
            + [f"gaus{n}" for n in range(1, 9, 1)]
            + [f"db{n}" for n in range(4, 21, 2)]
            + [f"sym{n}" for n in range(4, 21, 2)]
            + [f"coif{n}" for n in range(4, 17, 2)]
        )

        return np.array(wavelet_list)

    def ricker(self):
        """
        Генерирует вейвлет Рикера (вейвлет "мексиканская шляпа").

        Возвращает:
            ricker (numpy.ndarray): Вейвлет Рикера.
        """

        pi_fm_t = np.pi * self.dom_freq * (self.time - self.central_time_sec)
        ricker = (1 - 2 * (pi_fm_t) ** 2) * np.exp(-((pi_fm_t) ** 2))
        return ricker

    def morlet(self):
        """
        Генерирует вейвлет Морле.

        Возвращает:
            np.real(morlet_wavelet) (numpy.ndarray): действительная часть вейвлета Морле;
            np.imag(morlet_wavelet) (numpy.ndarray): мнимая часть вейвлета Морле;
            np.abs(morlet_wavelet) (numpy.ndarray): модуль (огибающая) вейвлета Морле.
        """

        # Формула для Морле:
        # Psi(t) = exp(-t^2 / (2 * sigma^2)) * exp(2 * pi * i * fm * t)

        # Вычисляем комплексный вейвлет
        morlet_wavelet = np.exp(
            -self.time_shifted**2 / (2 * self.sigma**2)
        ) * np.exp(  # огибающая (гауссиан)
            2j * np.pi * self.dom_freq * self.time_shifted
        )  # комплексная гармоническая составляющая

        # Возвращаем действительную, мнимую часть, а также модуль вейвлета
        return np.real(morlet_wavelet), np.imag(morlet_wavelet), np.abs(morlet_wavelet)

    def gaus(self, order):
        """
        Генерирует гауссов вейвлет заданного порядка.

        Параметры:
            order (int): Порядок производной (от 1 до 8).

        Возвращает:
            gauss_wavelet (numpy.ndarray): Гауссов вейвлет заданного порядка.
        """

        # Нормализуем время
        t_scaled = self.time_shifted / self.sigma

        # Получаем нормированный полином Ермита заданного порядка
        He_n = hermitenorm(order)(t_scaled)

        if order in [2, 3, 6, 7]:
            He_n = -He_n

        # Вычисляем коэффициент
        coeff = (-1) ** order / (self.sigma**order)

        # Вычисляем вейвлет
        gauss_wavelet = coeff * He_n * np.exp(-(t_scaled**2) / 2)

        return gauss_wavelet

    def daubechies(self, name, N):
        """
        Генерирует вейвлет Daubechies/Symlet/Coiflet указанного порядка с использованием каскадного алгоритма.

        Параметры:
            name (str): Принимает значения db/sym/coif (Daubechies, Symlets, Coiflet)
            N (int): Порядок вейвлета (4, 6, 8 ... 20 для db/sym, 4, 6, 8 ... 16 для coif).

        Возвращает:
            phi (numpy.ndarray): Функция масштабирования (вейвлет).
        """

        # if order not in h_coefficients:
        #     raise ValueError(f"Порядок вейвлета должен быть между 4 и 10, получено: {order}")

        h = self.symlet_coefficients(name, N)

        # Функция для увеличения дискретизации сигнала (вставка нулей между отсчетами)
        def upsample(signal):
            upsampled = np.zeros((len(signal) * 2) - 1)
            upsampled[::2] = signal
            return upsampled

        phi = np.array([1.0])  # Инициализация phi_0(t)

        # Выполнение итераций каскадного алгоритма для вычисления phi(t)
        for _ in range(self.num_cycles):
            # Увеличение дискретизации phi
            phi_up = upsample(phi)
            # Свертка с коэффициентами масштабирования h
            phi = np.sqrt(2) * np.convolve(phi_up, h, mode="full")

        # Вычисляем массив времени для phi, центрированный вокруг self.central_time_sec
        mid_index = len(phi) // 2
        t_phi = (
            np.arange(len(phi)) - mid_index
        ) * self.sample_rate_sec + self.central_time_sec

        # Интерполируем phi на массив времени self.time
        phi_resampled = np.interp(self.time, t_phi, phi)

        return phi_resampled

    def symlet_coefficients(self, name, N):
        """
        Выдает коэффициенты уравнения вейвлетов Добеши (dbN, symN, coifN)

        Параметры:
            name (str): Принимает значения db, sym, coif (Daubechies, Symlets, Coiflet)
            N (int): Порядок вейвлета.

        Возвращает:
            coef (numpy.ndarray): Коэффициенты уравнения.
        """

        key = f"{name}{N}"
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            data = {}
            with open(self.file_path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    # Пропускаем пустые строки и комментарии
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, value = line.split("=", 1)
                        k = k.strip()
                        value = value.strip()
                        try:
                            # Преобразуем строку JSON в список
                            data[k] = json.loads(value)
                        except json.JSONDecodeError:
                            raise ValueError(
                                f"Некорректный формат данных для ключа {k} в файле {self.file_path}"
                            )
            if key in data:
                return np.array(data[key])

        raise ValueError(f"Коэффициенты для {key} не найдены в файле {self.file_path}.")


if __name__ == "__main__":
    app = Wavelet("sym20")  # передаем название вейвлета
    graph = app.generate_wavelet()  # построение вейвлета
    plt.plot(graph)
