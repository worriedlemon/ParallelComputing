#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#include <time.h>

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

    int N = 1; // ������ ������ NxN

    while (N != 0)
    {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        std::vector<int> A(N * N), B(N * N), C(N * N);

        int rowsPerProcess = N / size; // ���������� ����� �� �������

        // ������������� ������ � �������� 0
        if (rank == 0) {
            for (int i = 0; i < N * N; ++i) {
                A[i] = rand() % 10; // ���������� ���������� ����������
                B[i] = rand() % 10;
            }
        }

        clock_t tStart = clock(); //������� ������� ����������

        // �������� ���� ��������� ������� B
        MPI_Bcast(B.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);

        // �������� ������ ��� ���������� A � ���������� ���������� C
        std::vector<int> local_A(rowsPerProcess * N);
        std::vector<int> local_C(rowsPerProcess * N);

        // ��������� ����� ������� A
        MPI_Scatter(A.data(), rowsPerProcess * N, MPI_INT, local_A.data(), rowsPerProcess * N, MPI_INT, 0, MPI_COMM_WORLD);

        // �������� ����������
        multiplyMatrices(local_A, B, local_C, N, rowsPerProcess);

        // �������� ����������
        MPI_Gather(local_C.data(), rowsPerProcess * N, MPI_INT, C.data(), rowsPerProcess * N, MPI_INT, 0, MPI_COMM_WORLD);

        // ������� ��������� �� ������� 0
        /*if (rank == 0) {
            std::cout << "result matrix c:\n";
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    std::cout << c[i * n + j] << " ";
                }
                std::cout << std::endl;
            }
        }*/

        printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC); //����� �������

        std::cout << "\nenter size matrix: ";
        std::cin >> N;
    }

    MPI_Finalize();
    return 0;
}
