#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <iomanip>

using namespace std;

// Параметры задачи
const int NUM_RUNS = 100; // Количество повторений

// Функция для решения СЛАУ методом Гаусса-Жордана с параллелизацией OpenMP
void gaussJordan(vector<vector<double>>& A, vector<double>& b, int num_threads) {
    int n = A.size();

    for (int i = 0; i < n; i++) {
        // Нормализуем строку i по диагональному элементу
        double diag = A[i][i];

#pragma omp parallel for num_threads(num_threads)
        for (int j = i; j < n; j++) {
            A[i][j] /= diag;
        }
        b[i] /= diag;

        // Обнуляем элементы в столбце i для всех строк, кроме строки i
#pragma omp parallel for num_threads(num_threads)
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = A[k][i];
                for (int j = i; j < n; j++) {
                    A[k][j] -= factor * A[i][j];
                }
                b[k] -= factor * b[i];
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
    cout << "Avg time: " << avg_time << " sec\r";
    cout.flush();  // Обновляем консоль
}

// Функция для вывода данных
void printResults(const vector<vector<double>>& A, const vector<double>& b, const vector<double>& result) {
    int n = A.size();
    cout << fixed << setprecision(2); // Установка формата вывода

    cout << "Initial matrix A:\n";
    for (const auto& row : A) {
        for (double val : row) {
            cout << setw(10) << val << "\t"; // Табуляция для лучшей читаемости
        }
        cout << endl;
    }

    cout << "\nInitial vector b:\n";
    for (double val : b) {
        cout << setw(10) << val << "\t"; // Табуляция для лучшей читаемости
    }
    cout << endl;

    cout << "\nResult vector x:\n";
    for (double val : result) {
        cout << setw(10) << val << "\t"; // Табуляция для лучшей читаемости
    }
    cout << endl;

    cout << std::resetiosflags(std::ios::fixed);
}

int main() {
    srand(time(0));  // Инициализация генератора случайных чисел
    int num_threads = 0;
    int N = -1;

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
            cin >> N;
            if (N == 0) {
                break;  // Повторный запрос ввода количества процессов
            }

            // Генерация случайной системы уравнений
            vector<vector<double>> A(N, vector<double>(N));
            vector<double> b(N);
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    A[i][j] = rand() % 100 + 1;  // Случайные числа от 1 до 100
                }
                b[i] = rand() % 100 + 1;
            }

            // Запрос вывода данных
            char display_choice;
            cout << "Do you want to display initial data and results? (Y/N): ";
            cin >> display_choice;

            // Приведение к верхнему регистру
            display_choice = toupper(display_choice);
            if (display_choice != 'Y' && display_choice != 'N') {
                cout << "Invalid input! Defaulting to 'N'." << endl;
                display_choice = 'N';
            }
            std::cout << "\033[A\033[2K";

            // Измерение времени выполнения с 100 повторениями
            double total_time = 0.0;

            vector<double> final_result(N);

            for (int i = 0; i < NUM_RUNS; i++) {
                // Создаем копии матрицы A и вектора b для каждого повторения
                vector<vector<double>> A_copy = A;
                vector<double> b_copy = b;

                double start_time = omp_get_wtime();  // Время начала
                gaussJordan(A_copy, b_copy, num_threads);
                double end_time = omp_get_wtime();    // Время завершения

                total_time += (end_time - start_time);  // Суммируем время

                // Отображение прогресса
                double avg_time = total_time / (i + 1);  // Среднее время выполнения на текущий момент
                displayProgress(i + 1, NUM_RUNS, avg_time);

                // Сохраняем последний результат
                final_result = b_copy; // После выполнения A_copy преобразуется в решение, сохраненное в b_copy
            }

            // Вывод окончательного результата
            std::cout << "\n\033[A\033[2K" << "Average execution time over " <<
                "\033[33m" << NUM_RUNS << "\033[0m"
                << " runs: " <<
                "\033[33m" << total_time / NUM_RUNS << "\033[0m"
                << " seconds." << std::endl;

            // Вывод данных, если пользователь выбрал Y
            if (display_choice == 'Y') {
                printResults(A, b, final_result);
            }
        }
    }

    return 0;
}
