import numpy as np
from math import ceil, floor
from scipy.special import hermitenorm
from scipy.interpolate import interp1d
from scipy.signal import convolve, get_window
from scipy.fft import fft, ifft, next_fast_len
from scipy.signal.windows import boxcar

class STFT:
    def __init__(self, segment_length, segment_length_padded, shift_length, window_function=boxcar):
        """
        Инициализирует объект класса STFT с заданными параметрами сегментации и оконной функции.

        Параметры:
        - segment_length: длина сегмента (целое число)
        - segment_length_padded: длина сегмента с дополнением нулями (целое число)
        - shift_length: длина сдвига (целое число)
        - window_function: функция окна (по умолчанию boxcar)
        """
        # Проверка типов переменных
        if not isinstance(segment_length, int) or segment_length <= 0:
            raise ValueError("segment_length должен быть положительным целым числом")
        if not isinstance(segment_length_padded, int) or segment_length_padded <= 0:
            raise ValueError("segment_length_padded должен быть положительным целым числом")
        if not isinstance(shift_length, int) or shift_length <= 0:
            raise ValueError("shift_length должен быть положительным целым числом")
        if not callable(window_function):
            raise ValueError("window_function должен быть вызываемой функцией")
        
        # Проверка соотношений между параметрами
        if shift_length > segment_length:
            raise ValueError("shift_length не должен превышать segment_length")
        if segment_length > segment_length_padded:
            raise ValueError("segment_length не должен превышать segment_length_padded")
        
        self.segment_length = segment_length
        self.segment_length_padded = segment_length_padded
        self.shift_length = shift_length
        self.window_function = window_function

    def compute_stft(self, x):
        """
        Вычисляет короткое преобразование Фурье (STFT) для входного сигнала x с использованием параметров, заданных в объекте класса.

        Параметры:
        - x: входной сигнал (numpy.ndarray)

        Возвращает:
        - x_stft: STFT преобразование сигнала x
        - start_list: список начальных индексов сегментов
        - stop_list: список конечных индексов сегментов
        """
        return self.stft(x, self.segment_length, self.segment_length_padded, self.shift_length, self.window_function)
    
    @staticmethod
    def stft(x, segment_length, segment_length_padded, shift_length, window_function):
        """
        Статический метод для вычисления STFT сигнала x с заданными параметрами.

        Параметры:
        - x: входной сигнал (numpy.ndarray)
        - segment_length: длина сегмента (целое число)
        - segment_length_padded: длина сегмента с дополнением нулями (целое число)
        - shift_length: длина сдвига (целое число)
        - window_function: функция окна

        Возвращает:
        - x_stft: STFT преобразование сигнала x
        - start_list: список начальных индексов сегментов
        - stop_list: список конечных индексов сегментов
        """
        
        if type(x) is not np.ndarray:
            raise ValueError("x is not numpy array")
        if segment_length > x.shape[0]:
            raise ValueError("segment_length is greater than x.shape[0]")
        if shift_length <= 0:
            raise ValueError("shift_length <= 0")
        if shift_length > segment_length:
            raise ValueError("shift_length > segment_length")
        if segment_length_padded < segment_length:
            raise ValueError("segment_length_padded < segment_length")

        window_vector = STFT.window_nonzero(window_function, segment_length)
        x_segments, start_list, stop_list = STFT.create_overlapping_segments(x, segment_length, shift_length)

        window_array = np.ones(x_segments.shape)
        for i in range(window_array.shape[0]):
            window_array[i] = window_array[i] * window_vector[i]
        
        x_segments = window_array * x_segments

        x_stft = np.fft.rfft(x_segments, n=segment_length_padded, axis=0)

        return x_stft, start_list, stop_list

    @staticmethod
    def window_nonzero(window_function, segment_length):
        """
        Статический метод для получения оконной функции без нулевых значений на концах.

        Параметры:
        - window_function: функция окна
        - segment_length: длина сегмента (целое число)

        Возвращает:
        - window_vector: вектор оконной функции без нулевых значений на концах
        """
        zero_exist = 1
        zero_count = 0
        
        window_vector = window_function(segment_length + zero_count)

        while zero_exist:            
            start = int(zero_count / 2)
            stop = int(len(window_vector) - zero_count / 2)
            window_vector = window_vector[start:stop]

            zero_count = len(window_vector) - np.count_nonzero(window_vector)
            
            if zero_count > 0:
                window_vector = window_function(segment_length + zero_count)
            else:
                zero_exist = 0

        return window_vector
    
    @staticmethod
    def create_overlapping_segments(x, segment_length, shift_length):
        """
        Статический метод для создания перекрывающихся сегментов сигнала.

        Параметры:
        - x: входной сигнал (numpy.ndarray)
        - segment_length: длина сегмента (целое число)
        - shift_length: длина сдвига (целое число)

        Возвращает:
        - x_segments: массив сегментов сигнала
        - start_list: список начальных индексов сегментов
        - stop_list: список конечных индексов сегментов
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("x не является numpy array")
        if segment_length > x.shape[0]:
            raise ValueError("segment_length больше длины сигнала x")
        if shift_length <= 0:
            raise ValueError("shift_length должно быть положительным")
        if shift_length > segment_length:
            raise ValueError("shift_length больше segment_length")
    
        x = np.squeeze(x)
        
        start_list = np.arange(0, x.shape[0], shift_length)
        stop_list = start_list + segment_length

        index = stop_list <= x.shape[0]
        start_list = start_list[index]
        stop_list = stop_list[index]

        if stop_list[-1] != x.shape[0]:
            stop_list = np.append(stop_list, x.shape[0])
            start_list = np.append(start_list, x.shape[0] - segment_length)

        x_segments = [x[start:stop] for start, stop in zip(start_list, stop_list)]
        x_segments = np.stack(x_segments, axis=1)

        return x_segments, start_list, stop_list

    @staticmethod
    def istft(x_stft, segment_length, segment_length_padded, start_list, stop_list,
              original_size, window_function, p):
        """
        Статический метод для вычисления обратного короткого преобразования Фурье (ISTFT).

        Параметры:
        - x_stft: STFT преобразование сигнала (numpy.ndarray)
        - segment_length: длина сегмента (целое число)
        - segment_length_padded: длина сегмента с дополнением нулями (целое число)
        - start_list: список начальных индексов сегментов
        - stop_list: список конечных индексов сегментов
        - original_size: исходный размер сигнала (кортеж или список)
        - window_function: функция окна
        - p: параметр степени окна (обычно p=2 для энергии)

        Возвращает:
        - x: восстановленный сигнал во временной области (numpy.ndarray)
        """
        # Выполняем обратное короткое преобразование Фурье
        x_segments = np.fft.irfft(x_stft, n=segment_length_padded, axis=0)
        x_segments = x_segments[:segment_length, :]

        # Генерируем оконную функцию
        window_vector = STFT.window_nonzero(window_function, segment_length)

        # Создаем массив окон той же размерности, что и x_segments
        # Это W^(p-1) в шаге 3 Таблицы 1 из источника [1]
        window_array = np.ones_like(x_segments)
        for i in range(window_array.shape[0]):
            window_array[i, :] = window_array[i, :] * window_vector[i]
        
        # Применяем оконную функцию к сегментам (window_array ** (p - 1))
        window_array = window_array ** (p - 1)
    
        # Применяем оконную функцию к сегментам
        x_segments = window_array * x_segments

        # Находим вектор наложения окон для операции оконного применения
        # Это Dp в шаге 5 Таблицы 1 из источника [1]
        window_overlap_add = np.zeros(original_size[0])
        number_segments = len(start_list)
        for i in range(number_segments):
            window_overlap_add[start_list[i]:stop_list[i]] += window_vector ** p
        
        # Инвертируем этот вектор
        window_overlap_add = window_overlap_add ** -1

        # Создаем выходной массив
        x = np.zeros(original_size)
        # Выполняем наложение и сложение сегментов
        for i, (start, stop) in enumerate(zip(start_list, stop_list)):
            x[start:stop, ...] += x_segments[:, i, ...]
        
        # Нормализуем x
        window_overlap_add_array = window_overlap_add[:, np.newaxis]

        x = x * window_overlap_add_array

        return x