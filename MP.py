import numpy as np
import matplotlib.pyplot as plt


class MatchingPursuit:
    """
    Класс для выполнения алгоритма Matching Pursuit и визуализации результатов с использованием распределения Вигнера-Вилля.

    Входные данные:
        trace (ndarray): Исходный сигнал для анализа.
        wavelets (list): Список вейвлетов для разложения.
        atoms (list): Список найденных атомов в виде (параметры, амплитуда).
    """

    def __init__(self, trace, wavelets):
        """
        Инициализация объекта MatchingPursuit.

        Параметры:
            trace (ndarray): Анализируемый сигнал.
            wavelets (list): Набор вейвлетов для разложения.
        """
        # Параметры по умолчанию
        self._sample_rate = 1  # Частота дискретизации (Гц)
        self._f_min = 0  # Минимальная частота для анализа (Гц)
        self._f_max = 100  # Максимальная частота для анализа (Гц)
        self._min_amplitude = 0.001  # Минимальная значимая амплитуда атома
        self._threshold = 0.001  # Порог остаточной энергии для остановки
        self._max_iterations = 3  # Максимальное число итераций алгоритма

        # Инициализация рабочих параметров
        self.num = len(wavelets)  # Количество вейвлетов
        self.wavelets = wavelets  # Нормализованные вейвлеты
        self.trace = trace  # Исходный сигнал
        self.wavelet_freqs = np.linspace(
            self._f_min, self._f_max, self.num
        )  # Частоты вейвлетов
        self.atoms = []  # Список найденных атомов

    # Блок свойств (properties) с валидацией значений
    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        if value <= 0:
            raise ValueError("Sample rate must be positive.")
        self._sample_rate = value

    @property
    def f_min(self):
        return self._f_min

    @f_min.setter
    def f_min(self, value):
        if value < 0:
            raise ValueError("Minimum frequency must be non-negative.")
        self._f_min = value
        self._update_wavelet_freqs()  # Обновляем wavelet_freqs при изменении f_min

    @property
    def f_max(self):
        return self._f_max

    @f_max.setter
    def f_max(self, value):
        if value <= self._f_min:
            raise ValueError(
                "Maximum frequency must be greater than minimum frequency."
            )
        self._f_max = value
        self._update_wavelet_freqs()  # Обновляем wavelet_freqs при изменении f_max

    @property
    def min_amplitude(self):
        return self._min_amplitude

    @min_amplitude.setter
    def min_amplitude(self, value):
        if value < 0:
            raise ValueError("Minimum amplitude must be non-negative.")
        self._min_amplitude = value

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        if value < 0:
            raise ValueError("Threshold must be non-negative.")
        self._threshold = value

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value):
        if not isinstance(value, int):  # Проверка, что значение является целым числом
            raise ValueError("max_iterations must be an integer.")
        if value <= 0:
            raise ValueError("Maximum iterations must be positive.")
        self._max_iterations = value

    def _update_wavelet_freqs(self):
        """Обновляет частоты вейвлетов при изменении f_min/f_max."""
        self.wavelet_freqs = np.linspace(self._f_min, self._f_max, self.num)

    def matching_pursuit(self, min_amplitude=0.001, threshold=0.001, max_iterations=3):
        """
        Алгоритм Matching Pursuit для разложения сигнала на атомы.

        Параметры:
            min_amplitude (float): Минимальная амплитуда для учета атома.
            threshold (float): Порог остаточной энергии для остановки.
            max_iterations (int): Максимальное число итераций.

        Возвращает:
            list: Список атомов в виде [((позиция, длина, частота), амплитуда), ...]
        """
        # Нормализация вейвлетов
        self.wavelets = [w / np.linalg.norm(w) for w in self.wavelets]
        wavelet_lengths = [len(w) for w in self.wavelets]

        tr = np.copy(self.trace)  # Копия сигнала для модификации
        self.atoms = []
        iter_count = 0
        residual_norm = np.linalg.norm(tr)  # Норма остатка

        # Основной цикл алгоритма
        while residual_norm > threshold and iter_count < max_iterations:
            # Вычисление сверток сигнала со всеми вейвлетами
            convs = np.array([np.convolve(tr, w, mode="same") for w in self.wavelets])

            # Поиск атома с максимальной корреляцией
            max_index = np.unravel_index(np.argmax(np.abs(convs)), convs.shape)
            a = convs[max_index]  # Амплитуда атома

            if np.abs(a) < min_amplitude:  # Проверка значимости амплитуды
                break

            # Извлечение параметров найденного атома
            wavelet_idx, pos = max_index
            L = wavelet_lengths[wavelet_idx]  # Длина вейвлета
            wavelet = self.wavelets[wavelet_idx]  # Форма вейвлета
            frequency = self.wavelet_freqs[wavelet_idx]  # Частота

            # Расчет границ влияния атома в сигнале
            i_s = pos - L // 2  # Начальный индекс
            i_f = i_s + L  # Конечный индекс

            # Корректировка границ при выходе за пределы массива
            cut_start = 0
            cut_end = 0
            if i_s < 0:
                cut_start = -i_s
                i_s = 0
            if i_f > len(tr):
                cut_end = i_f - len(tr)
                i_f = len(tr)

            # Вырезание значимой части вейвлета
            wavelet_segment = wavelet[cut_start : L - cut_end]

            # Вычитание вклада атома из сигнала
            tr[i_s:i_f] -= a * wavelet_segment

            # Сохранение параметров атома
            self.atoms.append(((pos, L, frequency), np.abs(a)))
            residual_norm = np.linalg.norm(tr)
            iter_count += 1

        print(f"Iterations performed: {iter_count}")
        return self.atoms

    def wigner_ville(self, params, t_grid, omega_grid):
        """
        Вычисление распределения Вигнера-Вилля для одного атома.

        Параметры:
            params (tuple): Параметры атома (время, ширина, частота).
            t_grid (ndarray): Временная сетка.
            omega_grid (ndarray): Частотная сетка.

        Возвращает:
            ndarray: Распределение Вигнера-Вилля для атома.
        """
        u, s, nu = params
        T, O = np.meshgrid(t_grid, omega_grid)
        # Формула псевдо-распределения Вигнера-Вилля для гауссова атома
        W = 2 * np.exp(-2 * np.pi * ((T - u) ** 2 / s**2 + (s**2 * (O - nu) ** 2)))
        return W

    def calculate_wigner_ville(self, atoms, t_grid, omega_grid):
        """
        Суммарное распределение Вигнера-Вилля для всех атомов.

        Параметры:
            atoms (list): Список атомов из matching_pursuit().
            t_grid (ndarray): Временная сетка.
            omega_grid (ndarray): Частотная сетка.

        Возвращает:
            ndarray: Суммарное распределение энергии.
        """
        W_sum = np.zeros((len(omega_grid), len(t_grid)))
        dt = t_grid[1] - t_grid[0] if len(t_grid) > 1 else 0

        for params, coeff in atoms:
            # Преобразование параметров атома
            atom_center, atom_width, atom_freq = params
            u = t_grid[atom_center]  # Центр атома во времени
            s = atom_width * dt  # Ширина атома в секундах
            nu = atom_freq  # Центральная частота

            # Расчет вклада атома и суммирование
            W = self.wigner_ville((u, s, nu), t_grid, omega_grid)
            W_sum += (np.abs(coeff) ** 2) * W  # Учет энергии атома

        return W_sum

    def plot_spectrum(self):
        """Визуализация спектральной плотности с помощью распределения Вигнера-Вилля."""
        atoms = self.matching_pursuit(
            self.min_amplitude, self.threshold, self.max_iterations
        )
        omega_grid = np.linspace(self.f_min, self.f_max, self.num)

        # Построение временной оси
        t = np.linspace(0, len(self.trace) * self.sample_rate / 1000, len(self.trace))

        # Расчет распределения
        W_sum = self.calculate_wigner_ville(atoms, t, omega_grid)

        # Настройка графика
        plt.figure(figsize=(12, 6))
        plt.imshow(
            W_sum,
            aspect="auto",
            origin="lower",
            extent=[t[0], t[-1], omega_grid[0], omega_grid[-1]],
            cmap="seismic",
        )  # Цветовая карта 'seismic'
        plt.colorbar(label="Энергия")
        plt.xlabel("Время, с")
        plt.ylabel("Частота, Гц")
        plt.title("MP")
        plt.show()


# Пример использования
signal = np.loadtxt("copy/signal.txt")  # Загрузка сигнала
wavelets_s = np.loadtxt("ricker_wavelets.txt")  # Загрузка вейвлетов Рикера
wavelets = [v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v for v in wavelets_s]

app = MatchingPursuit(signal, wavelets)
app.plot_spectrum()
