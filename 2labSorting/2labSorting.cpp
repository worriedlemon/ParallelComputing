#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>  // Для функции rand() и srand()
#include <ctime>    // Для функции time()

// Функция для сортировки методом вставок
void insertionSort(std::vector<int>& arr) {
    int n = arr.size();

    // Параллелизация с помощью OpenMP
#pragma omp parallel for shared(arr)
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;

        // Сдвигаем элементы arr[0..i-1], которые больше ключа,
        // на одну позицию вперед от их текущей позиции
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

int main() {

    int n = 1;

    while (n != 0) {
        // Инициализируем генератор случайных чисел
        srand(static_cast<unsigned>(time(0)));

        // Создаем массив и заполняем его случайными числами от 0 до n
        std::vector<int> arr(n);
        for (int i = 0; i < n; i++) {
            arr[i] = rand() % (n + 1);  // Генерируем случайное число от 0 до n
        }

        // Печать исходного массива
        /*std::cout << "source array: ";
        for (int i = 0; i < arr.size(); i++)
            std::cout << arr[i] << " ";
        std::cout << std::endl;*/

        clock_t tStart = clock(); // Подсчет времени выполнения

        // Выполняем сортировку
        insertionSort(arr);

        // Печать отсортированного массива
        /*std::cout << "sorted array: ";
        for (int i = 0; i < arr.size(); i++)
            std::cout << arr[i] << " ";
        std::cout << std::endl;*/

        // Вывод времени
        printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

        // Ввод размера массива
        std::cout << "enter array size: ";
        std::cin >> n;
    }

    return 0;
}