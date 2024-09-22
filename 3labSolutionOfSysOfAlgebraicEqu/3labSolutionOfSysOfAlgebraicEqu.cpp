#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

// ��������� ������
const double TOLERANCE = 1e-6; // ��������
const int MAX_ITERATIONS = 1000; // ������������ ���������� ��������

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
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x.data(), local_end - local_start, MPI_DOUBLE, MPI_COMM_WORLD);

        // ������� ���������� ����� (��������)
        double global_norm;
        MPI_Allreduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // �������� ����������
        if (sqrt(global_norm) < TOLERANCE) {
            if (rank == 0)
                std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 4; // ������ �������
    std::vector<std::vector<double>> A = {
        {4, 1, 2, 0},
        {3, 5, 1, 0},
        {1, 1, 3, 1},
        {2, 0, 1, 4}
    };

    std::vector<double> b = { 15, 28, 16, 21 };
    std::vector<double> x(N, 0.0); // ��������� �����������

    gauss_seidel_parallel(A, b, x, rank, size);

    // ����� ����������
    if (rank == 0) {
        std::cout << "Solution: ";
        for (int i = 0; i < N; ++i) {
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}
