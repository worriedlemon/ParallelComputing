#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// ������� ��� ������� ���� ������� ������-������� � ��������������� OpenMP
void gaussJordan(vector<vector<double>>& A, vector<double>& b) {
    int n = A.size();

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        // ������ ���: ������������ ������ �� ������������� ��������
        double diag = A[i][i];
#pragma omp parallel for
        for (int j = 0; j < n; j++) {
            A[i][j] /= diag;
        }
        b[i] /= diag;

        // ������ ���: ��������� ��������� ���� � ���� ���������
#pragma omp parallel for
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = A[k][i];
                for (int j = 0; j < n; j++) {
                    A[k][j] -= factor * A[i][j];
                }
                b[k] -= factor * b[i];
            }
        }
    }
}

int main() {
    // ������ �������������
    int n;
    cout << "enter size system: ";
    cin >> n;

    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);

    cout << "enter coeficient matrix A:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> A[i][j];
        }
    }

    cout << "enter vector b:\n";
    for (int i = 0; i < n; i++) {
        cin >> b[i];
    }

    // ������� �������
    gaussJordan(A, b);

    // ����� ����������
    cout << "solution system x:\n";
    for (int i = 0; i < n; i++) {
        cout << b[i] << " ";
    }
    cout << endl;

    return 0;
}
