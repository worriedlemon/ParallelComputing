#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>  // Для функции rand() и srand()
#include <ctime>    // Для функции time()
#include <algorithm> // Для std::min

// Параметры задачи
const int NUM_RUNS = 100; // Количество повторений

// Функция для сортировки методом вставок
// Сортирует часть массива от start до end
void insertionSort(std::vector<int>& arr, int start, int end) {
    for (int i = start + 1; i <= end; i++) {
        int key = arr[i];
        int j = i - 1;

        // Сдвигаем элементы, чтобы вставить ключ в нужное место
        while (j >= start && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Функция для слияния двух отсортированных блоков
// Сливает блоки массива между индексами left, mid и right
void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Создаем временные массивы для левого и правого блока
    std::vector<int> L(n1), R(n2);

    // Копируем данные в временные массивы
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int i = 0; i < n2; i++)
        R[i] = arr[mid + 1 + i];

    // Объединяем два блока обратно в основной массив
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Копируем оставшиеся элементы
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// Функция для параллельного слияния блоков
// Параллельно сливает блоки, увеличивая их размер в два раза
void mergeAllBlocks(std::vector<int>& arr, int num_threads, int block_size) {
    int n = arr.size();
    while (block_size < n) {
        // Параллельное выполнение слияния блоков
#pragma omp parallel for
        for (int i = 0; i < n; i += 2 * block_size) {
            int left = i;
            int mid = std::min(i + block_size - 1, n - 1);
            int right = std::min(i + 2 * block_size - 1, n - 1);

            // Если есть два блока, сливаем их
            if (mid < right) {
                merge(arr, left, mid, right);
            }
        }
        // Увеличиваем размер блоков для следующего слияния
        block_size *= 2;
    }
}

int main() {
    int num_threads;

    // Ввод количества потоков для параллельного выполнения
    std::cout << "Enter the number of threads: ";
    std::cin >> num_threads;

    // Устанавливаем количество потоков для OpenMP
    omp_set_num_threads(num_threads);

    int n;

    // Основной цикл программы
    while (true) {
        std::cout << "Enter array size (0 to change the number of threads): ";
        std::cin >> n;

        // Если 0, изменяем количество потоков
        if (n == 0) {
            std::cout << "Enter new number of threads: ";
            std::cin >> num_threads;
            omp_set_num_threads(num_threads);
            continue;
        }

        double total_time = 0.0;    // Время всех тестов

        // Выполняем несколько тестов для точности замеров
        for (int test = 0; test < NUM_RUNS; test++) {
            srand(static_cast<unsigned>(time(0)));  // Генерация случайного массива

            std::vector<int> arr(n);
            for (int i = 0; i < n; i++) {
                arr[i] = rand() % (n + 1);
            }

            // Засекаем начальное время
            double start_time = omp_get_wtime();

            // Определяем размер блока для каждого потока
            int block_size = (n + num_threads - 1) / num_threads;

            // Параллельная сортировка блоков методом вставок
#pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                int start = thread_id * block_size;
                int end = std::min(start + block_size - 1, n - 1);

                // Сортируем каждый блок параллельно
                if (start < n) {
                    insertionSort(arr, start, end);
                }
            }

            // Параллельное слияние отсортированных блоков
            mergeAllBlocks(arr, num_threads, block_size);

            // Засекаем конечное время
            double end_time = omp_get_wtime();

            // Суммируем время выполнения
            total_time += (end_time - start_time);
        }

        // Выводим результат
        std::cout << "Average sorting time for " <<
            "\033[33m" << n << "\033[0m"
            << " elements over " << NUM_RUNS << " tests: " <<
            "\033[33m" << total_time / NUM_RUNS << "\033[0m"
            << " seconds" << std::endl;
    }

    return 0;
}
