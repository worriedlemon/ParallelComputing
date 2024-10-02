#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

// Параметры задачи
const double TOLERANCE = 1e-6; // Точность
const int MAX_ITERATIONS = 1000; // Максимальное количество итераций
const int NUM_RUNS = 100; // Количество повторений

// Функция для генерации случайной матрицы и вектора
void generate_random_matrix_and_vector(std::vector<std::vector<double>>& A, std::vector<double>& b, int N) {
    srand(time(0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = rand() % 10 + 1; // случайное число от 1 до 10
        }
        A[i][i] += N * 10; // увеличиваем диагональные элементы для улучшения сходимости
        b[i] = rand() % 10 + 1; // случайное число от 1 до 10
    }
}

// Функция для решения СЛАУ методом Зейделя
void gauss_seidel_parallel(const std::vector<std::vector<double>>& A, const std::vector<double>& b, std::vector<double>& x, int rank, int size) {
    int N = A.size();
    int local_start = rank * N / size;   // Начало области, обрабатываемой процессом
    int local_end = (rank + 1) * N / size; // Конец области, обрабатываемой процессом

    std::vector<double> x_old(N, 0.0); // Вектор предыдущих значений

    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        double norm = 0.0;

        for (int i = local_start; i < local_end; ++i) {
            double sigma = 0.0;
            for (int j = 0; j < N; ++j) {
                if (j != i)
                    sigma += A[i][j] * x[j];
            }

            x_old[i] = x[i]; // Сохраняем старое значение
            x[i] = (b[i] - sigma) / A[i][i]; // Обновляем значение
            norm += std::pow(x[i] - x_old[i], 2);
        }

        // Обмен данными между процессами
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x.data(), N / size, MPI_DOUBLE, MPI_COMM_WORLD);
        //
        // Эта строка обменивает обновленные части решения (вектор x) между всеми процессами. 
        // Каждый процесс сначала решает свою часть задачи, 
        // потом они обмениваются результатами, 
        // чтобы все знали об обновлениях.
        //

        // Считаем глобальную норму (точность)
        double global_norm;
        MPI_Allreduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        //
        // Процессы складывают свои локальные ошибки (нормы) и получают общую ошибку, 
        // чтобы проверить, насколько решение близко к правильному.
        //

        // Проверка сходимости
        if (sqrt(global_norm) < TOLERANCE) {
            break;
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

    while (true) {
        int N;

        // Ввод размера системы
        std::cout << "Enter the size of the system (N), or 0 to exit: ";
        std::cin >> N;

        // Распространяем значение N на все процессы
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // 
        // используются для того, чтобы главный процесс (с рангом 0)
        // разослал всем остальным процессам данные — размер матрицы. 
        // Это нужно, чтобы каждый процесс имел одинаковую копию данных и мог с ними работать.)
        //

        if (N == 0) break; // Выход из цикла, если введено 0

        // Инициализация матрицы и вектора
        std::vector<std::vector<double>> A(N, std::vector<double>(N));
        std::vector<double> b(N);
        std::vector<double> x(N, 0.0); // Начальное приближение

        // Генерируем случайную матрицу и вектор
        generate_random_matrix_and_vector(A, b, N);

        // Вывод начальных значений опционально
        //std::cout << "Initial system:" << std::endl;
        //for (int i = 0; i < N; ++i) {
        //    for (int j = 0; j < N; ++j) {
        //        std::cout << A[i][j] << " ";
        //    }
        //    std::cout << " | " << b[i] << std::endl;
        //}

        // Распространяем матрицу и вектор b на все процессы
        for (int i = 0; i < N; ++i) {
            MPI_Bcast(A[i].data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            // 
            // используются для того, чтобы главный процесс (с рангом 0)
            // разослал всем остальным процессам данные — данные матрицы. 
            // Это нужно, чтобы каждый процесс имел одинаковую копию данных и мог с ними работать.)
            //
        }
      
        MPI_Bcast(b.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // 
        // используются для того, чтобы главный процесс (с рангом 0)
        // разослал всем остальным процессам данные — данные вектора. 
        // Это нужно, чтобы каждый процесс имел одинаковую копию данных и мог с ними работать.)
        //

        // Переменная для хранения общего времени
        double total_time = 0.0;

        // Выполняем решение 100 раз
        for (int run = 0; run < NUM_RUNS; ++run) {
            // Замер времени с использованием MPI
            double start_time = MPI_Wtime();

            // Решаем СЛАУ методом Зейделя
            gauss_seidel_parallel(A, b, x, rank, size);

            // Замер времени завершен
            double end_time = MPI_Wtime();

            // Суммируем время выполнения
            total_time += (end_time - start_time);
        }

        // Вывод результата
        // Выводим решение (после последнего запуска) опционально
        //std::cout << "Solution: ";
        //for (int i = 0; i < N; ++i) {
        //    std::cout << x[i] << " ";
        //}
        //std::cout << std::endl;

        // Вычисляем и выводим среднее время выполнения
        double average_time = total_time / NUM_RUNS;
        std::cout << "Average execution time over " <<
            "\033[33m" << NUM_RUNS << "\033[0m"
            << " runs: " <<
            "\033[33m" << average_time << "\033[0m"
            << " seconds." << std::endl;
    }

    MPI_Finalize();
    //
    // В конце программы MPI_Finalize(); завершает работу MPI и освобождает все ресурсы.
    //
    return 0;
}