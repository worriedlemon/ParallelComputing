#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <chrono>
#include <mpi.h>

using namespace std;

// Функция для вычисления разложения Холецкого для блока
void cholesky_block(vector<vector<double>>& A, int start_row, int block_size) {
    for (int k = 0; k < block_size; ++k) {
        // Диагональный элемент
        A[start_row + k][start_row + k] = sqrt(A[start_row + k][start_row + k]);

        // Ниже диагонального элемента
        for (int i = k + 1; i < block_size; ++i) {
            A[start_row + i][start_row + k] /= A[start_row + k][start_row + k];
        }

        // Обновление остальной части матрицы
        for (int j = k + 1; j < block_size; ++j) {
            for (int i = j; i < block_size; ++i) {
                A[start_row + i][start_row + j] -= A[start_row + i][start_row + k] * A[start_row + j][start_row + k];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;

    while (true) {
        if (rank == 0) {
            cout << "Enter the size of the matrix (0 to exit): ";
            cin >> n;

            if (n == 0) {
                MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
                break;
            }
        }

        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (n == 0) {
            break;
        }

        // Инициализация матрицы A на процессе 0
        vector<vector<double>> A(n, vector<double>(n));
        if (rank == 0) {
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> dist(0.0, 1.0);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    A[i][j] = dist(gen);
                }
                A[i][i] += n; // Для положительной определенности
            }
        }

        // Распространение A на все процессы
        for (int i = 0; i < n; i++)
            MPI_Bcast(&A[i][0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);


        int block_size = n / size;  // Размер блока (предполагается, что n делится на size)

        auto start_time = chrono::high_resolution_clock::now();

        for (int k = 0; k < size; ++k) {
            if (rank == k) {
                cholesky_block(A, k * block_size, block_size);
                // Рассылка вычисленного блока
                for (int i = 0; i < block_size; i++)
                    MPI_Bcast(&A[k * block_size + i][0], n, MPI_DOUBLE, k, MPI_COMM_WORLD);

            }
            else {
                // Прием вычисленного блока
                for (int i = 0; i < block_size; i++)
                    MPI_Bcast(&A[k * block_size + i][0], n, MPI_DOUBLE, k, MPI_COMM_WORLD);
            }

            // Обновление остальных блоков (эта часть пока не распараллелена)
            if (rank > k) {
                for (int i = (k + 1) * block_size; i < n; i++) {
                    for (int j = k * block_size; j < (k + 1) * block_size; ++j) {
                        // L_ik = (A_ik - sum(L_ij*L_kj, j=0..k-1)) / L_kk
                        double sum = 0.0;
                        for (int l = 0; l < block_size; l++) {
                            sum += A[i][k * block_size + l] * A[k * block_size + j][k * block_size + l];

                        }
                        A[i][j] = (A[i][j] - sum) / A[j][j];
                    }
                }
            }
        }


        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> diff = end_time - start_time;

        if (rank == 0) {
            cout << "Cholesky decomposition completed in " << diff.count() << " s." << endl;

            // Вывод матрицы A (результат разложения)
            char output_A;
            cout << "Do you want to display the result (matrix A)? (Y/N): ";
            cin >> output_A;
            if (toupper(output_A) == 'Y') {
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        cout << setw(10) << A[i][j] << " "; // Выводим A, а не L
                    }
                    cout << endl;
                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}