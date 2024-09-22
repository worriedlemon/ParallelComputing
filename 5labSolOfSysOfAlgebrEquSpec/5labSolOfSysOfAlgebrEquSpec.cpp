#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>

using namespace std;

void chol_decomp(int n, vector<vector<double>>& A, vector<vector<double>>& L, int rank, int size) {
    for (int i = 0; i < n; i++) {
        if (i % size == rank) {
            for (int j = 0; j <= i; j++) {
                double sum = 0;
                for (int k = 0; k < j; k++)
                    sum += L[i][k] * L[j][k];

                if (i == j)
                    L[i][j] = sqrt(A[i][i] - sum);
                else
                    L[i][j] = (A[i][j] - sum) / L[j][j];
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 4; // размерность матрицы
    vector<vector<double>> A = { {25, 15, -5, -10},
                                {15, 18,  0,  -6},
                                {-5,  0, 11,   7},
                                {-10, -6, 7,  11} }; // симметричная матрица
    vector<vector<double>> L(n, vector<double>(n, 0));
    vector<double> b = { 1, 2, 3, 4 }; // правая часть СЛАУ

    // Разложение Холецкого
    chol_decomp(n, A, L, rank, size);

    // Решение систем L * y = b и L^T * x = y
    vector<double> y = forward_substitution(L, b);
    vector<double> x = backward_substitution(L, y);

    // Вывод решения
    if (rank == 0) {
        cout << "Solution x: ";
        for (int i = 0; i < n; i++) {
            cout << x[i] << " ";
        }
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}
