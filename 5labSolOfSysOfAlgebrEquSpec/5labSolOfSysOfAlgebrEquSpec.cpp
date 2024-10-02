#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

// ��������� ������
const int NUM_RUNS = 10; // ���������� ����������

// ������� ��� ��������� ��������� ������������ ������������ ����������� �������
vector<vector<double>> generate_positive_definite_matrix(int n) {
    vector<vector<double>> A(n, vector<double>(n));

    // ������ ��������� ������������ �������
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            A[i][j] = rand() % 10 + 1; // ��������� ����� �� 1 �� 10
            A[j][i] = A[i][j];
        }
        // ��������� ������������ �������� ��� ����������� ������������� �������������
        A[i][i] += n; // ����������� ������������ ��������
    }

    return A;
}

// ��������� ���������� �������
vector<double> generate_random_vector(int n) {
    vector<double> b(n);
    for (int i = 0; i < n; i++) {
        b[i] = rand() % 10 + 1; // ��������� ����� �� 1 �� 10
    }
    return b;
}

// ������� ��� ��������, �������� �� ������� ������������ �����������
bool is_positive_definite(const vector<vector<double>>& A) {
    int n = A.size();
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++) {
            sum += A[i][j] * A[i][j];
        }
        // ���������, ��� ������������ ������� �����������
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

        // ������������� ������ ����� ����������
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

// ������� ��� ����������� ��������� ����������
void displayProgress(int current, int total, double avg_time) {
    int barWidth = 50;  // ������ ��������-����
    double progress = (double)current / total;  // �������� ���������� � �����

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
    cout.flush();  // ��������� �������
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(0) + rank); // ������������� ���������� ���������� ��� ������� ��������

    int n;
    while (true) {
        cout << "Enter the size of the matrix (0 to exit): ";
        cin >> n;
        if (n == 0) {
            // �������� ���� ���������, ��� ����� �����������
            for (int p = 1; p < size; p++) {
                MPI_Send(&n, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
            }
            break;
        }

        // �������� ������� ������� ���� ���������
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        double total_time = 0;

        for (int i = 0; i < NUM_RUNS; i++) {
            vector<vector<double>> A;
            do {
                // ��������� ��������� ������������ ����������� �������
                A = generate_positive_definite_matrix(n);
            } while (!is_positive_definite(A)); // �������� �� ������������� �������������

            vector<vector<double>> L(n, vector<double>(n, 0));
            vector<double> b = generate_random_vector(n);

            // ��������� ������� � ������� MPI
            double start_time = MPI_Wtime();

            // ���������� ���������
            chol_decomp(n, A, L, rank, size);

            // ������� ������ L * y = b � L^T * x = y
            vector<double> y = forward_substitution(L, b);
            vector<double> x = backward_substitution(L, y);

            double end_time = MPI_Wtime();
            total_time += (end_time - start_time);

            // ����������� ���������
            double avg_time = total_time / (i + 1);  // ������� ����� ���������� �� ������� ������
            displayProgress(i + 1, NUM_RUNS, avg_time);
        }

        // ���������� � ����� �������� ������� ����������
        std::cout << endl << "Average execution time over " <<
            "\033[33m" << NUM_RUNS << "\033[0m"
            << " runs: " <<
            "\033[33m" << total_time / NUM_RUNS << "\033[0m"
            << " seconds." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
