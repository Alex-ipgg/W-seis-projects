class CWT:
    def __init__(self, scales=None, frequencies=None, wavelet='gaus1', sampling_period=1.0, method='fft', axis=-1, bfreq=500):
        # Проверка параметра 'bfreq'
        if not isinstance(bfreq, (int, float)):
            raise TypeError("Параметр 'bfreq' должен быть числом.")
        if bfreq <= 0:
            raise ValueError("Параметр 'bfreq' должен быть положительным числом.")
        
        # Проверка параметра 'frequencies' и 'scales'
        if frequencies is not None:
            frequencies = np.atleast_1d(frequencies)
            if not np.issubdtype(frequencies.dtype, np.number):
                raise TypeError("Параметр 'frequencies' должен быть числовым массивом.")
            if np.any(frequencies <= 0):
                raise ValueError("Частоты должны быть положительными значениями.")
            self.scales = bfreq / (np.sqrt(2.0) * np.pi * frequencies)
            if np.any(self.scales <= 0):
                raise ValueError("Вычисленные масштабы должны быть положительными значениями.")
        elif scales is not None:
            scales = np.atleast_1d(scales)
            if not np.issubdtype(scales.dtype, np.number):
                raise TypeError("Параметр 'scales' должен быть числовым массивом.")
            if np.any(scales <= 0):
                raise ValueError("Масштабы должны быть положительными значениями.")
            self.scales = scales
        else:
            raise ValueError("Необходимо предоставить либо 'scales', либо 'frequencies'.")

        # Проверка параметра 'wavelet'
        if not isinstance(wavelet, str):
            raise TypeError("Параметр 'wavelet' должен быть строкой.")
        # Здесь можно добавить дополнительную проверку на допустимые значения для wavelet
        self.wavelet = Wavelet(wavelet).generate_wavelet()

        # Проверка параметра 'sampling_period'
        if not isinstance(sampling_period, (int, float)):
            raise TypeError("Параметр 'sampling_period' должен быть числом.")
        if sampling_period <= 0:
            raise ValueError("Параметр 'sampling_period' должен быть положительным числом.")
        self.sampling_period = sampling_period

        # Проверка параметра 'method'
        if method not in ['fft', 'conv']:
            raise ValueError("Параметр 'method' должен быть либо 'fft', либо 'conv'.")
        self.method = method

        # Проверка параметра 'axis'
        if not isinstance(axis, int):
            raise TypeError("Параметр 'axis' должен быть целым числом.")
        self.axis = axis

        # Инициализация остальных параметров
        self.int_psi, self.x = self._prepare_wavelet()

        
    def _prepare_wavelet(self, precision=10):
        """
        Подготовка вейвлет-функции для свертки.
    
        Параметры:
        - precision: int, необязательный
            Точность для генерации вейвлета. По умолчанию 10.
    
        Возвращает:
        - int_psi: ndarray
            Массив значений вейвлет-функции.
        - x: ndarray
            Соответствующие позиции для значений вейвлета.
        """
        int_psi = np.asarray(self.wavelet)
        x = np.linspace(-5, 5, int_psi.size)
        return int_psi, x
    
    def _compute_conv_result(self, data, int_psi_scale):
        """
        Вычисление результата свертки с использованием указанного метода.
    
        Параметры:
        - data: ndarray
            Входной массив данных.
        - int_psi_scale: ndarray
            Масштабированная вейвлет-функция.
    
        Возвращает:
        - output: ndarray
            Результат свертки.
        """
        if self.method == 'conv':
            # Используем np.apply_along_axis для векторизации свертки
            output = np.apply_along_axis(
                lambda m: convolve(m, int_psi_scale, mode='full'), axis=-1, arr=data
            )
        elif self.method == 'fft':
            conv_size = next_fast_len(data.shape[-1] + int_psi_scale.size - 1)
    
            # Выполняем быстрое преобразование Фурье по последней оси
            fft_data = fft(data, n=conv_size, axis=-1)
            fft_wavelet = fft(int_psi_scale, n=conv_size)
    
            # Умножаем в частотной области и возвращаем в временную
            conv = ifft(fft_data * fft_wavelet, axis=-1)
    
            # Обрезаем до нужного размера и берем действительную часть
            output = conv[..., :data.shape[-1] + int_psi_scale.size - 1].real
        else:
            raise ValueError("Method must be either 'conv' or 'fft'.")
        return output

    def compute(self, data):
        """
        Вычисление непрерывного вейвлет-преобразования (CWT) для заданных данных.
    
        Параметры:
        - data: array_like
            Входной массив данных.
    
        Возвращает:
        - out: ndarray
            Массив коэффициентов CWT.
        """
        data = np.asarray(data)
        dt = data.dtype
    
        # Определяем тип данных для комплексных вычислений
        dt_cplx = np.result_type(dt, np.complex128)
        dt_out = dt_cplx
    
        original_shape = data.shape
        axis = self.axis
    
        if data.ndim > 1:
            # Перемещаем целевую ось в конец и приводим данные к двумерному массиву
            data = np.swapaxes(data, axis, -1)
            data_shape = data.shape
            data = data.reshape(-1, data_shape[-1])
        else:
            data = data[np.newaxis, :]
    
        num_scales = len(self.scales)
        num_signals, signal_length = data.shape
        out_shape = (num_scales, num_signals, signal_length)
        out = np.empty(out_shape, dtype=dt_out)
    
        # Предварительные вычисления
        step = self.x[1] - self.x[0]
        x_range = self.x[-1] - self.x[0]
        len_int_psi = len(self.int_psi)
    
        for i, scale in enumerate(self.scales):
            # Вычисляем индексы для int_psi_scale
            N = int(scale * x_range) + 1
            j = np.linspace(0, N - 1, N) / (scale * step)
            j = j.astype(int)
            j = j[j < len_int_psi]
    
            # Получаем масштабированную функцию int_psi
            int_psi_scale = self.int_psi[j][::-1]
    
            # Вычисляем свертку
            conv_result = self._compute_conv_result(data, int_psi_scale)
    
            # Вычисляем коэффициенты CWT
            coef = -np.sqrt(scale) * np.diff(conv_result, axis=-1)
    
            if dt_out.kind != 'c':
                coef = coef.real
    
            # Корректируем длину коэффициентов
            excess_length = coef.shape[-1] - signal_length
            if excess_length > 0:
                trim_start = excess_length // 2
                trim_end = trim_start + excess_length % 2
                coef = coef[:, trim_start:-trim_end]
    
            out[i] = coef
    
        # Восстанавливаем исходную форму данных
        out = out.reshape((num_scales,) + original_shape)
        if data.ndim > 1:
            out = np.swapaxes(out, -1, axis)
    
        return out.squeeze()


    def scale2frequency(self, precision=10):
        """
        Преобразование масштаба в частоты на основе вейвлет-функции.
    
        Параметры:
        - precision: int, необязательно
            Точность для преобразования частоты. Значение по умолчанию - 10.
    
        Возвращается:
        - frequencies: ndarray
            Массив частот, соответствующих шкалам.
        """
        frequencies = scale2frequency(self.wavelet, self.scales, precision)
        frequencies = np.atleast_1d(frequencies)
        return frequencies / self.sampling_period