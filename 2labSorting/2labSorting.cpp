#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>  // Для функции rand() и srand()
#include <ctime>    // Для функции time()

// Функция для сортировки методом вставок
void insertionSort(std::vector<int>& arr, int start, int end) {
    for (int i = start + 1; i <= end; i++) {
        int key = arr[i];
        int j = i - 1;

        while (j >= start && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

int main() {
    int num_threads;

    // Ввод числа потоков один раз в начале
    std::cout << "Enter threads count: ";
    std::cin >> num_threads;

    // Установка числа потоков
    omp_set_num_threads(num_threads);

    int n;

    while (true) {
        // Ввод размера массива
        std::cout << "Enter array size (0 for edit threads count): ";
        std::cin >> n;

        if (n == 0) {
            // Изменение числа потоков
            std::cout << "Enter threads count: ";
            std::cin >> num_threads;
            omp_set_num_threads(num_threads);
            continue;  // Возврат к началу цикла для нового ввода размера массива
        }

        double total_time = 0;
        int test_count = 100;

        // Выполнение 100 тестов
        for (int test = 0; test < test_count; test++) {
            // Инициализируем генератор случайных чисел
            srand(static_cast<unsigned>(time(0)));

            // Создаем массив и заполняем его случайными числами от 0 до n
            std::vector<int> arr(n);
            for (int i = 0; i < n; i++) {
                arr[i] = rand() % (n + 1);  // Генерируем случайное число от 0 до n
            }

            // Подсчет времени выполнения сортировки с использованием OpenMP
            double start_time = omp_get_wtime();

            // Размер блока
            int block_size = n / num_threads;

            // Параллельная сортировка блоков
#pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                int start = thread_id * block_size;
                int end = (thread_id == num_threads - 1) ? n - 1 : (start + block_size - 1);

                insertionSort(arr, start, end);
            }

            // На данном этапе можно использовать стратегию слияния блоков для полной сортировки массива

            double end_time = omp_get_wtime();

            // Суммируем время выполнения
            total_time += (end_time - start_time);
        }

        // Вычисление среднего времени выполнения
        double avg_time = total_time / (float)test_count;

        // Вывод среднего времени выполнения сортировки
        std::cout << "Average time taken " << n << " elements: " << avg_time << " seconds" << std::endl;
    }

    return 0;
}
