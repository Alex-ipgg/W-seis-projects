import time
import scipy
import numpy as np
import matplotlib.pyplot as plt


class MatchingPursuit:
    """
    Класс для выполнения алгоритма Matching Pursuit и визуализации результатов с использованием распределения Вигнера-Вилля.
    """
    def __init__(self, trace, wavelets, norm = False):
        """
        Инициализация объекта MatchingPursuit.
        Параметры:
        trace (ndarray): Анализируемый сигнал.
        wavelets (list): Набор вейвлетов для разложения.
        """
        # Параметры по умолчанию
        self._sampling_interval = 2  # Интервал дискретизации (мс)
        self._wavelet_min_freq = 0  # Минимальная частота вейвлета (Гц)
        self._wavelet_max_freq = 100  # Максимальная частота вейвлета (Гц)
        self._min_amplitude = 0.1  # Минимальная значимая амплитуда атома
        self._residual_energy_threshold = 0.1  # Порог остаточной энергии для остановки
        self._max_iterations = 100  # Максимальное число итераций алгоритма
        # Параметры сглаживания
        self._affine_smoothing_mu = 1
        self._time_smoothing_sigma = 1  # Сглаживание по времени
        self._frequency_smoothing_sigma = 1  # Сглаживание по частоте

        # Инициализация рабочих параметров
        self.num = len(wavelets)  # Количество вейвлетов
        self.wavelets = [w / np.linalg.norm(w) for w in wavelets]
        self.max_amplitudes = [np.max(np.abs(w)) for w in self.wavelets]  # Сохраняем максимальные амплитуды
        
        if norm:
            self.trace = trace / max(abs(trace))
        else:
            self.trace = trace  # Исходный сигнал
            
        self.signal_length = len(self.trace)
        self.wavelet_length = len(self.wavelets[0])
        self.wavelet_freqs = np.linspace(self._wavelet_min_freq, self._wavelet_max_freq, self.num)
        self.t_grid = np.linspace(0, self.signal_length * self.sampling_interval / 1000, self.signal_length)
        self.reflection_coefficients = []  # Коэффициенты отражения
    
    @property
    def sampling_interval(self):
        return self._sampling_interval

    @sampling_interval.setter
    def sampling_interval(self, value):
        if value <= 0:
            raise ValueError("Интервал дискретизации должен быть положительным.")
        self._sampling_interval = value
        self._update_time_grid()

    @property
    def wavelet_min_freq(self):
        return self._wavelet_min_freq

    @wavelet_min_freq.setter
    def wavelet_min_freq(self, value):
        if value < 0:
            raise ValueError("Минимальная частота вейвлета должна быть неотрицательной.")
        self._wavelet_min_freq = value
        self._update_wavelet_freqs()

    @property
    def wavelet_max_freq(self):
        return self._wavelet_max_freq

    @wavelet_max_freq.setter
    def wavelet_max_freq(self, value):
        if value <= self._wavelet_min_freq:
            raise ValueError("Максимальная частота вейвлета должна превышать минимальную.")
        self._wavelet_max_freq = value
        self._update_wavelet_freqs()

    @property
    def min_amplitude(self):
        return self._min_amplitude

    @min_amplitude.setter
    def min_amplitude(self, value):
        if value < 0:
            raise ValueError("Минимальная амплитуда должна быть неотрицательной.")
        self._min_amplitude = value

    @property
    def residual_energy_threshold(self):
        return self._residual_energy_threshold

    @residual_energy_threshold.setter
    def residual_energy_threshold(self, value):
        if value < 0:
            raise ValueError("Порог остаточной энергии должен быть неотрицательным.")
        self._residual_energy_threshold = value

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value):
        if not isinstance(value, int):
            raise ValueError("Максимальное число итераций должно быть целым числом.")
        if value <= 0:
            raise ValueError("Максимальное число итераций должно быть положительным.")
        self._max_iterations = value

    @property
    def time_smoothing_sigma(self):
        return self._time_smoothing_sigma

    @time_smoothing_sigma.setter
    def time_smoothing_sigma(self, value):
        if value < 0:
            raise ValueError("Сглаживание по времени должно быть положительным.")
        self._time_smoothing_sigma = value

    @property
    def frequency_smoothing_sigma(self):
        return self._frequency_smoothing_sigma

    @frequency_smoothing_sigma.setter
    def frequency_smoothing_sigma(self, value):
        if value < 0:
            raise ValueError("Сглаживание по частоте должно быть положительным.")
        self._frequency_smoothing_sigma = value

    @property
    def affine_smoothing_mu(self):
        return self._affine_smoothing_mu

    @affine_smoothing_mu.setter
    def affine_smoothing_mu(self, value):
        if value < 0:
            raise ValueError("Коэффициент аффинного сглаживания должен быть положительным.")
        self._affine_smoothing_mu = value
    
    def _update_wavelet_freqs(self):
        """Обновляет частоты вейвлетов при изменении минимальной/максимальной частоты."""
        self.wavelet_freqs = np.linspace(self._wavelet_min_freq, self._wavelet_max_freq, self.num)

    def _update_time_grid(self):
        """Обновляет временную сетку при изменении интервала дискретизации."""
        self.t_grid = np.linspace(0, self.signal_length * self._sampling_interval / 1000, self.signal_length)

    def matching_pursuit_with_wigner_ville(self):
        """
        Алгоритм Matching Pursuit с расчетом распределения Вигнера-Вилля и сбором коэффициентов отражения.
        """
        residual_signal = np.copy(self.trace)  # Остаточный сигнал
        wigner_ville_distribution = np.zeros((self.num, self.signal_length), dtype=np.float64)  # Распределение Вигнера-Вилля
        residual_norm = np.linalg.norm(residual_signal)  # Норма остаточного сигнала
        iteration_count = 0  # Счетчик итераций
        self.reflection_coefficients = []  # Коэффициенты отражения
    
        while residual_norm > self._residual_energy_threshold and iteration_count < self._max_iterations:
            # Свертка остаточного сигнала с вейвлетами
            convolutions = np.array([np.convolve(residual_signal, wavelet, mode="same") for wavelet in self.wavelets])
            max_conv_index = np.unravel_index(np.argmax(np.abs(convolutions)), convolutions.shape)
            amplitude = convolutions[max_conv_index]
    
            if np.abs(amplitude) < self._min_amplitude:
                break
    
            wavelet_index, position = max_conv_index
            selected_wavelet = self.wavelets[wavelet_index]
            frequency = self.wavelet_freqs[wavelet_index]
    
            # Сохранение коэффициентов отражения
            self.reflection_coefficients.append({
                'amplitude': amplitude,
                'time_position': position,
                'frequency': frequency,
                'wavelet_index': wavelet_index
            })
    
            # Обновление остаточного сигнала
            start_index = max(0, position - self.wavelet_length // 2)
            end_index = min(len(residual_signal), start_index + self.wavelet_length)
            residual_signal[start_index:end_index] -= amplitude * selected_wavelet[:end_index - start_index]
    
            # Расчет распределения Вигнера-Вилля для текущего атома
            time_grid, frequency_grid = np.meshgrid(self.t_grid, self.wavelet_freqs)
            wigner_ville_atom = 2 * np.exp(-2 * np.pi * (
                ((time_grid - self.t_grid[position]) ** 2 / (self.wavelet_length * (self.t_grid[1] - self.t_grid[0])) ** 2) +
                ((self.wavelet_length * (self.t_grid[1] - self.t_grid[0])) ** 2 * (frequency_grid - frequency) ** 2)
            )) * (np.abs(amplitude) ** 2)
    
            # Сглаживание и добавление к общему распределению
            smoothed_wigner_ville_atom = self._affine_smoothing(wigner_ville_atom)
            wigner_ville_distribution += smoothed_wigner_ville_atom
    
            residual_norm = np.linalg.norm(residual_signal)
            iteration_count += 1
    
        return wigner_ville_distribution

    def _affine_smoothing(self, W):
        """
        Применяет аффинное сглаживание к распределению Вигнера-Вилля.
        """
        sigma_t = self.time_smoothing_sigma
        mu = self.affine_smoothing_mu
        sigma_f = 1 / (mu ** 2 * sigma_t)
        smoothed_W = scipy.ndimage.gaussian_filter(W, sigma=(sigma_f, sigma_t))
        return smoothed_W

    def plot_reflection_coefficients(self):
        """
        Визуализирует трассу коэффициентов отражения.
        
        Параметры:
        ax (matplotlib.axes.Axes): Ось для отрисовки. Если None, создается новая фигура.
        show (bool): Флаг немедленного отображения графика.
        
        Возвращает:
        time_series (np.array): трасса коэффициентов отражения
        """
        if not self.reflection_coefficients:
            raise ValueError("Сначала выполните разложение методом matching_pursuit_with_wigner_ville()")
    
        time_series = np.zeros(self.signal_length)
        
        for coeff in self.reflection_coefficients:
            pos = coeff['time_position']
            wavelet_idx = coeff['wavelet_index']
            scaled_amplitude = coeff['amplitude'] * self.max_amplitudes[wavelet_idx]
            if 0 <= pos < self.signal_length:
                # Сохраняем амплитуду с наибольшим абсолютным значением в текущей позиции
                if abs(scaled_amplitude) > abs(time_series[pos]):
                    time_series[pos] = scaled_amplitude
    
        return time_series

# Пример использования
signal = np.loadtxt("copy/signal.txt")  # Загрузка сигнала
# signal = np.loadtxt(r"C:\Users\ALEKS\Downloads\Telegram Desktop\S_vector.txt")
wavelets_s = np.loadtxt("ricker_wavelets.txt")  # Загрузка вейвлетов Рикера
wavelets = [v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v for v in wavelets_s]

start_time = time.time()
app = MatchingPursuit(signal, wavelets, norm = True)
app.sample_rate = 2
W_sum = app.matching_pursuit_with_wigner_ville()
end_time = time.time()
total_time = end_time - start_time
print(f"Общее время выполнения: {total_time:.5f} секунд")

# Временная шкала
t_grid = np.linspace(0, len(signal) * app.sample_rate / 1000, len(signal))
wavelet_freqs = np.linspace(0, 100, 100)

# Получение коэффициентов
coef = app.plot_reflection_coefficients()

# Построение графика
plt.figure(figsize=(12, 8))

# График Wigner-Ville
plt.subplot(2, 1, 1)
plt.imshow(
    W_sum,
    aspect="auto",
    origin="lower",
    extent=[t_grid[0], t_grid[-1], wavelet_freqs[0], wavelet_freqs[-1]],
    cmap="hot",
)
plt.colorbar(label="Энергия")
plt.xlabel("Время, с")
plt.ylabel("Частота, Гц")
plt.title(f"MP")

# График коэффициентов
plt.subplot(2, 1, 2)
plt.stem(t_grid, coef, linefmt='b-', markerfmt='bo', basefmt=" ")
plt.xlabel("Время, с")
plt.ylabel("Коэффициенты")
plt.title("Коэффициенты отражения")

plt.tight_layout()
plt.show()
