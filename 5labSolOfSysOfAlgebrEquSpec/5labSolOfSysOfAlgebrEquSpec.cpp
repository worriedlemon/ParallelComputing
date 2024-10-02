#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <numeric> // ���������� ��� ������������� inner_product

using namespace std;

// ������� ��� ���������� ���������
void cholesky_decomposition(const vector<vector<double>>& A, vector<vector<double>>& L) {
    int n = A.size();

    for (int j = 0; j < n; j++) {
        // ���������� ������������� ��������
        double sum = (j > 0) ? inner_product(L[j].begin(), L[j].begin() + j, L[j].begin(), 0.0) : 0.0;
        L[j][j] = sqrt(A[j][j] - sum);

        for (int i = j + 1; i < n; i++) {
            // ���������� �������� ���� ���������
            double sumL = inner_product(L[i].begin(), L[i].begin() + j, L[j].begin(), 0.0);
            L[i][j] = (A[i][j] - sumL) / L[j][j];
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // ������������� MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // ��������� ������ ��������
    MPI_Comm_size(MPI_COMM_WORLD, &size); // ��������� ������ ����� ���������

    int n = 0;
    vector<vector<double>> A;

    while (true) {
        // ������ ������� ������� �� �������� 0
        if (rank == 0) {
            cout << "Enter the size of the matrix (0 to exit, max size = 10000): ";
            cin >> n;

            if (n == 0) {
                cout << "Exiting the program." << endl;
                MPI_Finalize(); // ���������� MPI ����� �������
                return 0;
            }

            // ����������� �� ������������ ������ �������
            if (n > 10000) {
                cout << "Size exceeds maximum limit of 10000." << endl;
                continue;
            }

            // ��������� ��������� ������������ ������������ �������
            A.resize(n, vector<double>(n));
            srand(static_cast<unsigned int>(time(0)));

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    A[i][j] = static_cast<double>(rand()) / RAND_MAX; // ��������� �������� �� 0 �� 1
                }
            }

            // ����������� ������������� �������������� �������
            for (int i = 0; i < n; i++) {
                A[i][i] += n; // ����������� ������������ ��������
            }
        }

        // ��������������� ������� ������� ����� ���� ���������
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // ��������� ������, ���� n ������ ���������� ���������
        if (size > n) {
            if (rank == 0) {
                cout << "Number of processes exceeds the size of the matrix. Adjusting number of processes." << endl;
            }
            size = n; // ������������� ���������� ��������� ������ ������� �������
            MPI_Comm_size(MPI_COMM_WORLD, &size); // ���������� ���������� ���������
        }

        // ������������� ������� A ����� ���������
        int rows_per_process = (n + size - 1) / size; // ���������� ����� �� ������� (� ������ �������)
        vector<vector<double>> A_local(rows_per_process, vector<double>(n));

        // ���������, ��� ���������� ������ �� ��������� ������ ������������ �������
        vector<double> flat_A(A.size() * A[0].size());
        if (rank == 0) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    flat_A[i * n + j] = A[i][j];
                }
            }
        }

        // ������������� ������� A ����� ���������
        MPI_Scatter(flat_A.data(), rows_per_process * n, MPI_DOUBLE, A_local.data(), rows_per_process * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // ���������� ���������� ���������
        vector<vector<double>> L_local(rows_per_process, vector<double>(n, 0.0));
        double start_time = MPI_Wtime(); // ������ �������

        cholesky_decomposition(A_local, L_local); // ����� ������� ����������

        double end_time = MPI_Wtime(); // ��������� �������

        // ���� ����������� ���������� �� ���� ���������
        vector<vector<double>> L(n, vector<double>(n, 0.0));
        MPI_Gather(L_local.data(), rows_per_process * n, MPI_DOUBLE, L.data(), rows_per_process * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // ����� ����������� �� �������� 0
        if (rank == 0) {
            // ���������� ������� ����������
            double total_time = end_time - start_time;

            // ��������� �������� �� ������������ �������
            if (total_time < 0) {
                cout << "Warning: Negative execution time detected." << endl;
            }

            cout << "Total execution time: " << total_time << " seconds" << endl;

            // ������ �� ����� �������� ������
            char output_A, output_L;
            cout << "Do you want to display the original matrix A? (Y/N): ";
            cin >> output_A;

            cout << "Do you want to display the result of the decomposition (matrix L)? (Y/N): ";
            cin >> output_L;

            // ����� �������� ������� A, ���� ������������ �����
            if (toupper(output_A) == 'Y') {
                cout << "Original matrix A:" << endl;
                for (const auto& row : A) {
                    for (const auto& elem : row) {
                        cout << setw(10) << elem << " ";
                    }
                    cout << endl;
                }
            }

            // ����� ���������� ���������� L, ���� ������������ �����
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

    MPI_Finalize(); // ���������� ������ MPI
    return 0;
}
