#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// Параметры задачи
const int NUM_RUNS = 10; // Количество повторений

// Функция для решения СЛАУ методом Гаусса-Жордана с параллелизацией OpenMP
void gaussJordanParallel(vector<vector<double>>& A, vector<double>& b, int num_threads) {
    int n = A.size();

    // Устанавливаем количество потоков для OpenMP
    omp_set_num_threads(num_threads);

    // Прямой ход метода Гаусса
    for (int k = 0; k < n; ++k) {
        // Нормализуем ведущий элемент на строке
#pragma omp parallel for
        for (int j = k + 1; j < n; ++j) {
            A[k][j] /= A[k][k];
        }
        b[k] /= A[k][k];
        A[k][k] = 1.0;

        // Прямой ход исключения по другим строкам
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            if (i != k) {
                double factor = A[i][k];
                for (int j = k + 1; j < n; ++j) {
                    A[i][j] -= factor * A[k][j];
                }
                b[i] -= factor * b[k];
                A[i][k] = 0.0;
            }
        }
    }
}

// Функция для отображения прогресса выполнения
void displayProgress(int current, int total, double avg_time) {
    int barWidth = 50;  // Ширина прогресс-бара
    double progress = (double)current / total;  // Прогресс выполнения в долях

    cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << int(progress * 100.0) << "% ";
    cout << "Completed: " << current << "/" << total << " ";
    cout << "Avg time: " << "\033[33m" << avg_time << "\033[0m" << " sec\r";
    cout.flush();  // Обновляем консоль
}

int main() {
    srand(time(0));  // Инициализация генератора случайных чисел
    int num_threads = 0;
    int n = -1;

    // Ввод количества процессов и обработка выхода из программы при нуле
    while (true) {
        cout << "Enter number of threads (0 to exit): ";
        cin >> num_threads;
        if (num_threads == 0) {
            cout << "Program terminated.\n";
            return 0;
        }

        // Ввод размерности системы
        while (true)
        {
            cout << "Enter system size (0 to re-enter number of threads): ";
            cin >> n;
            if (n == 0) {
                break;  // Повторный запрос ввода количества процессов
            }

            // Генерация случайной системы уравнений
            vector<vector<double>> A(n, vector<double>(n));
            vector<double> b(n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    A[i][j] = rand() % 100 + 1;  // Случайные числа от 1 до 100
                }
                b[i] = rand() % 100 + 1;
            }

            // Измерение времени выполнения с 100 повторениями
            double total_time = 0.0;

            for (int i = 0; i < NUM_RUNS; i++) {
                // Создаем копии матрицы A и вектора b для каждого повторения
                vector<vector<double>> A_copy = A;
                vector<double> b_copy = b;

                double start_time = omp_get_wtime();  // Время начала
                gaussJordanParallel(A_copy, b_copy, num_threads);
                double end_time = omp_get_wtime();    // Время завершения

                total_time += (end_time - start_time);  // Суммируем время

                // Отображение прогресса
                double avg_time = total_time / (i + 1);  // Среднее время выполнения на текущий момент
                displayProgress(i + 1, NUM_RUNS, avg_time);
            }

            // Вывод окончательного результата
            std::cout << "\n\033[A\033[2K" << "Average execution time over " <<
                "\033[33m" << NUM_RUNS << "\033[0m"
                << " runs: " <<
                "\033[33m" << total_time / NUM_RUNS << "\033[0m"
                << " seconds." << std::endl;
        }
    }

    return 0;
}
