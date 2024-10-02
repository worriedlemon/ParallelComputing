#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

// Параметры задачи
const int NUM_RUNS = 10; // Количество повторений

// Функция для генерации случайной симметричной положительно определённой матрицы
vector<vector<double>> generate_positive_definite_matrix(int n) {
    vector<vector<double>> A(n, vector<double>(n));

    // Создаём случайную симметричную матрицу
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            A[i][j] = rand() % 10 + 1; // случайное число от 1 до 10
            A[j][i] = A[i][j];
        }
        // Добавляем диагональные элементы для обеспечения положительной определённости
        A[i][i] += n; // Увеличиваем диагональные элементы
    }

    return A;
}

// Генерация случайного вектора
vector<double> generate_random_vector(int n) {
    vector<double> b(n);
    for (int i = 0; i < n; i++) {
        b[i] = rand() % 10 + 1; // случайное число от 1 до 10
    }
    return b;
}

// Функция для проверки, является ли матрица положительно определённой
bool is_positive_definite(const vector<vector<double>>& A) {
    int n = A.size();
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++) {
            sum += A[i][j] * A[i][j];
        }
        // Проверяем, что диагональный элемент положителен
        if (A[i][i] - sum <= 0) {
            return false;
        }
    }
    return true;
}

void chol_decomp(int n, vector<vector<double>>& A, vector<vector<double>>& L, int rank, int size) {
    for (int i = 0; i < n; i++) {
        if (i % size == rank) {
            for (int j = 0; j <= i; j++) {
                double sum = 0;
                for (int k = 0; k < j; k++)
                    sum += L[i][k] * L[j][k];

                if (i == j) {
                    L[i][j] = sqrt(A[i][i] - sum);
                }
                else {
                    L[i][j] = (A[i][j] - sum) / L[j][j];
                }
            }
        }

        // Синхронизация данных между процессами
        for (int p = 0; p < size; ++p) {
            MPI_Bcast(&L[i][0], i + 1, MPI_DOUBLE, p, MPI_COMM_WORLD);
        }
    }
}

vector<double> forward_substitution(const vector<vector<double>>& L, const vector<double>& b) {
    int n = L.size();
    vector<double> y(n);
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }
    return y;
}

vector<double> backward_substitution(const vector<vector<double>>& L, const vector<double>& y) {
    int n = L.size();
    vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < n; j++) {
            sum += L[j][i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i][i];
    }
    return x;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(0) + rank); // Инициализация случайного генератора для каждого процесса

    int n;
    while (true) {
        cout << "Enter the size of the matrix (0 to exit): ";
        cin >> n;
        if (n == 0) {
            // Сообщаем всем процессам, что нужно завершиться
            for (int p = 1; p < size; p++) {
                MPI_Send(&n, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
            }
            break;
        }

        // Рассылка размера матрицы всем процессам
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        double total_time = 0;

        for (int i = 0; i < NUM_RUNS; i++) {
            vector<vector<double>> A;
            do {
                // Генерация случайной положительно определённой матрицы
                A = generate_positive_definite_matrix(n);
            } while (!is_positive_definite(A)); // Проверка на положительную определённость

            vector<vector<double>> L(n, vector<double>(n, 0));
            vector<double> b = generate_random_vector(n);

            // Измерение времени с помощью MPI
            double start_time = MPI_Wtime();

            // Разложение Холецкого
            chol_decomp(n, A, L, rank, size);

            // Решение систем L * y = b и L^T * x = y
            vector<double> y = forward_substitution(L, b);
            vector<double> x = backward_substitution(L, y);

            double end_time = MPI_Wtime();
            total_time += (end_time - start_time);

            // Отображение прогресса
            double avg_time = total_time / (i + 1);  // Среднее время выполнения на текущий момент
            displayProgress(i + 1, NUM_RUNS, avg_time);
        }

        // Вычисление и вывод среднего времени выполнения
        std::cout << endl << "Average execution time over " <<
            "\033[33m" << NUM_RUNS << "\033[0m"
            << " runs: " <<
            "\033[33m" << total_time / NUM_RUNS << "\033[0m"
            << " seconds." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
