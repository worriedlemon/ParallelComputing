#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <numeric> // Подключаем для использования inner_product

using namespace std;

// Функция для разложения Холецкого
void cholesky_decomposition(const vector<vector<double>>& A, vector<vector<double>>& L) {
    int n = A.size();

    for (int j = 0; j < n; j++) {
        // Вычисление диагонального элемента
        double sum = (j > 0) ? inner_product(L[j].begin(), L[j].begin() + j, L[j].begin(), 0.0) : 0.0;
        L[j][j] = sqrt(A[j][j] - sum);

        for (int i = j + 1; i < n; i++) {
            // Вычисление элемента ниже диагонали
            double sumL = inner_product(L[i].begin(), L[i].begin() + j, L[j].begin(), 0.0);
            L[i][j] = (A[i][j] - sumL) / L[j][j];
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Инициализация MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получение номера процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Получение общего числа процессов

    int n = 0;
    vector<vector<double>> A;

    while (true) {
        // Запрос размера матрицы на процессе 0
        if (rank == 0) {
            cout << "Enter the size of the matrix (0 to exit): ";
            cin >> n;

            if (n == 0) {
                cout << "Exiting the program." << endl;
                MPI_Finalize(); // Завершение MPI перед выходом
                return 0;
            }

            // Генерация случайной положительно определенной матрицы
            A.resize(n, vector<double>(n));
            srand(static_cast<unsigned int>(time(0)));

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    A[i][j] = static_cast<double>(rand()) / RAND_MAX; // Случайные значения от 0 до 1
                }
            }

            // Обеспечение положительной определенности матрицы
            for (int i = 0; i < n; i++) {
                A[i][i] += n; // Увеличиваем диагональные элементы
            }
        }

        // Распространение размера матрицы среди всех процессов
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Обработка случая, если n меньше количества процессов
        if (size > n) {
            if (rank == 0) {
                cout << "Number of processes exceeds the size of the matrix. Adjusting number of processes." << endl;
            }
            size = n; // Устанавливаем количество процессов равным размеру матрицы
            MPI_Comm_size(MPI_COMM_WORLD, &size); // Обновление количества процессов
        }

        // Распределение матрицы A среди процессов
        int rows_per_process = (n + size - 1) / size; // Количество строк на процесс (с учетом остатка)
        vector<vector<double>> A_local(rows_per_process, vector<double>(n));

        // Убедитесь, что выделяемый размер не превышает размер оригинальной матрицы
        vector<double> flat_A(A.size() * A[0].size());
        if (rank == 0) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    flat_A[i * n + j] = A[i][j];
                }
            }
        }

        // Распределение матрицы A среди процессов
        MPI_Scatter(flat_A.data(), rows_per_process * n, MPI_DOUBLE, A_local.data(), rows_per_process * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Выполнение разложения Холецкого
        vector<vector<double>> L_local(rows_per_process, vector<double>(n, 0.0));
        double start_time = MPI_Wtime(); // Запуск таймера

        cholesky_decomposition(A_local, L_local); // Вызов функции разложения

        double end_time = MPI_Wtime(); // Остановка таймера

        // Сбор результатов разложения от всех процессов
        vector<vector<double>> L(n, vector<double>(n, 0.0));
        MPI_Gather(L_local.data(), rows_per_process * n, MPI_DOUBLE, L.data(), rows_per_process * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Вывод результатов на процессе 0
        if (rank == 0) {
            // Вычисление времени выполнения
            double total_time = end_time - start_time;

            // Добавляем проверку на корректность времени
            if (total_time < 0) {
                cout << "Warning: Negative execution time detected." << endl;
            }

            cout << "Total execution time: " << total_time << " seconds" << endl;

            // Запрос на вывод исходных данных
            char output_A, output_L;
            cout << "Do you want to display the original matrix A? (Y/N): ";
            cin >> output_A;

            cout << "Do you want to display the result of the decomposition (matrix L)? (Y/N): ";
            cin >> output_L;

            // Вывод исходной матрицы A, если пользователь хочет
            if (toupper(output_A) == 'Y') {
                cout << "Original matrix A:" << endl;
                for (const auto& row : A) {
                    for (const auto& elem : row) {
                        cout << setw(10) << elem << " ";
                    }
                    cout << endl;
                }
            }

            // Вывод результата разложения L, если пользователь хочет
            if (toupper(output_L) == 'Y') {
                cout << "Result of the decomposition (matrix L):" << endl;
                for (const auto& row : L) {
                    for (const auto& elem : row) {
                        cout << setw(10) << elem << " ";
                    }
                    cout << endl;
                }
            }
        }
    }

    MPI_Finalize(); // Завершение работы MPI
    return 0;
}
