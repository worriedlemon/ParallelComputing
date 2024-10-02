#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

void multiplyMatrices(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, int N, int rowsPerProcess) {
    for (int i = 0; i < rowsPerProcess; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    
    MPI_Init(&argc, &argv);
    //
    // В начале программы строка MPI_Init(&argc, &argv); 
    // запускает MPI, 
    // чтобы все процессы (то есть разные ядра) могли начать общаться между собой.
    //

    int N = 1; // Размер матриц NxN

    while (N != 0) {
        int rank, size;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        //
        // MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
        // присваивает каждому процессу уникальный номер (ранг), 
        // чтобы каждый процесс знал, какой именно частью работы он занимается.
        //

        MPI_Comm_size(MPI_COMM_WORLD, &size);
        //
        // MPI_Comm_size(MPI_COMM_WORLD, &size); 
        // говорит программе, сколько всего процессов участвует в расчёте.
        //

        std::vector<int> A(N * N), B(N * N), C(N * N);

        int rowsPerProcess = N / size; // количество строк на процесс

        // Инициализация матриц в процессе 0
        if (rank == 0) {
            for (int i = 0; i < N * N; ++i) {
                A[i] = rand() % 10; // Заполнение случайными значениями
                B[i] = rand() % 10;
            }
        }

        // Начало замера общего времени с помощью MPI_Wtime
        double start_time = MPI_Wtime();

        // Передаем всем процессам матрицу B
        MPI_Bcast(B.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
        // 
        // используются для того, чтобы главный процесс (с рангом 0)
        // разослал всем остальным процессам данные — матрица B. 
        // Это нужно, чтобы каждый процесс имел одинаковую копию данных и мог с ними работать.)
        //

        // Выделяем память для подматрицы A и результата подматрицы C
        std::vector<int> local_A(rowsPerProcess * N);
        std::vector<int> local_C(rowsPerProcess * N);

        // Рассылаем части матрицы A
        MPI_Scatter(A.data(), rowsPerProcess * N, MPI_INT, local_A.data(), rowsPerProcess * N, MPI_INT, 0, MPI_COMM_WORLD);
        //
        // Рассылаем части матрицы A от процесса 0 к другим процессам. 
        // Каждый процесс получает свою часть строк матрицы.
        //

        // Умножаем подматрицу
        multiplyMatrices(local_A, B, local_C, N, rowsPerProcess);

        // Собираем результаты
        MPI_Gather(local_C.data(), rowsPerProcess * N, MPI_INT, C.data(), rowsPerProcess * N, MPI_INT, 0, MPI_COMM_WORLD);
        // 
        // Собираем результаты умножения обратно на процесс 0.
        // 

        // Завершаем замер времени
        double end_time = MPI_Wtime();

        // Выводим общее время выполнения только на процесс 0
        if (rank == 0) {
            std::cout << "Total time taken: " << (end_time - start_time) << " seconds\n";
        }

        // Получаем новый размер матрицы от пользователя
        if (rank == 0) {
            std::cout << "\nEnter size matrix (0 to exit): ";
            std::cin >> N;
        }

        // Передаем размер матрицы всем процессам
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // 
        // используются для того, чтобы главный процесс (с рангом 0)
        // разослал всем остальным процессам данные — новый размер матрицы. 
        // Это нужно, чтобы каждый процесс имел одинаковую копию данных и мог с ними работать.)
        //
    }

    MPI_Finalize();
    return 0;
}
