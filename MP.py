import numpy as np
import matplotlib.pyplot as plt

class MatchingPursuit:
    """
    Класс для выполнения алгоритма Matching Pursuit и визуализации результатов с использованием распределения Вигнера-Вилля.
    Входные данные:
    trace (ndarray): Исходный сигнал для анализа.
    wavelets (list): Список вейвлетов для разложения.
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
        self.wavelets = [w / np.linalg.norm(w) for w in wavelets]
        self.trace = trace  # Исходный сигнал
        self.lenth = len(self.trace)
        self.wavelet_lenth = len(self.wavelets[0])
        self.wavelet_freqs = np.linspace(self._f_min, self._f_max, self.num)  # Частоты вейвлетов
        self.t_grid = np.linspace(0, self.lenth * self.sample_rate / 1000, self.lenth)

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        if value <= 0:
            raise ValueError("Sample rate must be positive.")
        self._sample_rate = value
        self._update_time_grid()
        
    @property
    def f_min(self):
        return self._f_min

    @f_min.setter
    def f_min(self, value):
        if value < 0:
            raise ValueError("Minimum frequency must be non-negative.")
        self._f_min = value
        self._update_wavelet_freqs()

    @property
    def f_max(self):
        return self._f_max

    @f_max.setter
    def f_max(self, value):
        if value <= self._f_min:
            raise ValueError("Maximum frequency must be greater than minimum frequency.")
        self._f_max = value
        self._update_wavelet_freqs()

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
        if not isinstance(value, int):
            raise ValueError("max_iterations must be an integer.")
        if value <= 0:
            raise ValueError("Maximum iterations must be positive.")
        self._max_iterations = value

    def _update_wavelet_freqs(self):
        """Обновляет частоты вейвлетов при изменении f_min/f_max."""
        self.wavelet_freqs = np.linspace(self._f_min, self._f_max, self.num)

    def _update_time_grid(self):
        """Обновляет время при изменении sample_rate."""
        self.t_grid = np.linspace(0, self.lenth * self._sample_rate / 1000, self.lenth)

    def _wigner_ville(self, params, omega_grid):
        """
        Вычисление распределения Вигнера-Вилля для одного атома.
        """
        u, s, nu = params
        T, O = np.meshgrid(self.t_grid, omega_grid)
        return 2 * np.exp(-2 * np.pi * ((T - u) ** 2 / s**2 + (s**2 * (O - nu) ** 2)))

    def matching_pursuit_with_wigner_ville(self):
        """
        Алгоритм Matching Pursuit с одновременным расчетом распределения Вигнера-Вилля.
        """
        tr = np.copy(self.trace)
        W_sum = np.zeros((len(self.wavelet_freqs), len(self.t_grid)))
        residual_norm = np.linalg.norm(tr)
        iter_count = 0

        while residual_norm > self._threshold and iter_count < self._max_iterations:
            # Вычисление сверток сигнала со всеми вейвлетами
            convs = np.array([np.convolve(tr, w, mode="same") for w in self.wavelets])
            max_index = np.unravel_index(np.argmax(np.abs(convs)), convs.shape)
            a = convs[max_index]

            if np.abs(a) < self._min_amplitude:
                break

            # Извлечение параметров найденного атома
            wavelet_idx, pos = max_index
            wavelet = self.wavelets[wavelet_idx]
            frequency = self.wavelet_freqs[wavelet_idx]

            # Корректировка границ влияния атома
            i_s = max(0, pos - self.wavelet_lenth // 2)
            i_f = min(len(tr), i_s + self.wavelet_lenth)
            wavelet_segment = wavelet[:i_f - i_s]

            # Вычитание вклада атома из сигнала
            tr[i_s:i_f] -= a * wavelet_segment

            # Расчет распределения Вигнера-Вилля для текущего атома
            W = self._wigner_ville(
                (self.t_grid[pos], self.wavelet_lenth * (self.t_grid[1] - self.t_grid[0]), frequency),
                self.wavelet_freqs
            )
            W_sum += (np.abs(a) ** 2) * W

            # Обновление нормы остатка
            residual_norm = np.linalg.norm(tr)
            iter_count += 1

        print(f"Iterations performed: {iter_count}")
        return W_sum

# Пример использования
signal = np.loadtxt("copy/signal.txt")  # Загрузка сигнала
wavelets_s = np.loadtxt("ricker_wavelets.txt")  # Загрузка вейвлетов Рикера
wavelets = [v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v for v in wavelets_s]

app = MatchingPursuit(signal, wavelets)
W_sum = app.matching_pursuit_with_wigner_ville()

sample_rate = 1
t_grid = np.linspace(0, len(signal) * sample_rate / 1000, len(signal))
wavelet_freqs = np.linspace(0, 100, 100)

# Настройка графика
plt.figure(figsize=(12, 6))
plt.imshow(
    W_sum,
    aspect="auto",
    origin="lower",
    extent=[t_grid[0], t_grid[-1], wavelet_freqs[0], wavelet_freqs[-1]],
    cmap="seismic",
)
plt.colorbar(label="Энергия")
plt.xlabel("Время, с")
plt.ylabel("Частота, Гц")
plt.title("MP")
plt.show()
