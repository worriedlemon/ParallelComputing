#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

// ��������� ������
const double TOLERANCE = 1e-6; // ��������
const int MAX_ITERATIONS = 1000; // ������������ ���������� ��������
const int NUM_RUNS = 100; // ���������� ����������

// ������� ��� ��������� ��������� ������� � �������
void generate_random_matrix_and_vector(std::vector<std::vector<double>>& A, std::vector<double>& b, int N) {
    srand(time(0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = rand() % 10 + 1; // ��������� ����� �� 1 �� 10
        }
        A[i][i] += N * 10; // ����������� ������������ �������� ��� ��������� ����������
        b[i] = rand() % 10 + 1; // ��������� ����� �� 1 �� 10
    }
}

// ������� ��� ������� ���� ������� �������
void gauss_seidel_parallel(const std::vector<std::vector<double>>& A, const std::vector<double>& b, std::vector<double>& x, int rank, int size) {
    int N = A.size();
    int local_start = rank * N / size;   // ������ �������, �������������� ���������
    int local_end = (rank + 1) * N / size; // ����� �������, �������������� ���������

    std::vector<double> x_old(N, 0.0); // ������ ���������� ��������

    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        double norm = 0.0;

        for (int i = local_start; i < local_end; ++i) {
            double sigma = 0.0;
            for (int j = 0; j < N; ++j) {
                if (j != i)
                    sigma += A[i][j] * x[j];
            }

            x_old[i] = x[i]; // ��������� ������ ��������
            x[i] = (b[i] - sigma) / A[i][i]; // ��������� ��������
            norm += std::pow(x[i] - x_old[i], 2);
        }

        // ����� ������� ����� ����������
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x.data(), N / size, MPI_DOUBLE, MPI_COMM_WORLD);

        // ������� ���������� ����� (��������)
        double global_norm;
        MPI_Allreduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // �������� ����������
        if (sqrt(global_norm) < TOLERANCE) {
            break;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    while (true) {
        int N;

        // ���� ������� �������
        std::cout << "Enter the size of the system (N), or 0 to exit: ";
        std::cin >> N;

        // �������������� �������� N �� ��� ��������
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (N == 0) break; // ����� �� �����, ���� ������� 0

        // ������������� ������� � �������
        std::vector<std::vector<double>> A(N, std::vector<double>(N));
        std::vector<double> b(N);
        std::vector<double> x(N, 0.0); // ��������� �����������

        // ���������� ��������� ������� � ������
        generate_random_matrix_and_vector(A, b, N);

        // ����� ��������� �������� �����������
        //std::cout << "Initial system:" << std::endl;
        //for (int i = 0; i < N; ++i) {
        //    for (int j = 0; j < N; ++j) {
        //        std::cout << A[i][j] << " ";
        //    }
        //    std::cout << " | " << b[i] << std::endl;
        //}

        // �������������� ������� � ������ b �� ��� ��������
        for (int i = 0; i < N; ++i) {
            MPI_Bcast(A[i].data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        MPI_Bcast(b.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // ���������� ��� �������� ������ �������
        double total_time = 0.0;

        // ��������� ������� 100 ���
        for (int run = 0; run < NUM_RUNS; ++run) {
            // ����� ������� � �������������� MPI
            double start_time = MPI_Wtime();

            // ������ ���� ������� �������
            gauss_seidel_parallel(A, b, x, rank, size);

            // ����� ������� ��������
            double end_time = MPI_Wtime();

            // ��������� ����� ����������
            total_time += (end_time - start_time);
        }

        // ����� ����������
        // ������� ������� (����� ���������� �������) �����������
        //std::cout << "Solution: ";
        //for (int i = 0; i < N; ++i) {
        //    std::cout << x[i] << " ";
        //}
        //std::cout << std::endl;

        // ��������� � ������� ������� ����� ����������
        double average_time = total_time / NUM_RUNS;
        std::cout << "Average execution time over " <<
            "\033[33m" << NUM_RUNS << "\033[0m"
            << " runs: " <<
            "\033[33m" << average_time << "\033[0m"
            << " seconds." << std::endl;
    }

    MPI_Finalize();
    return 0;
}