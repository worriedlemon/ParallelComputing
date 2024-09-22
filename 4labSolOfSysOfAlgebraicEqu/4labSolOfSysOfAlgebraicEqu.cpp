#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Функция для решения СЛАУ методом Гаусса-Жордана с параллелизацией OpenMP
void gaussJordan(vector<vector<double>>& A, vector<double>& b) {
    int n = A.size();

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        // Прямой ход: нормализация строки по диагональному элементу
        double diag = A[i][i];
#pragma omp parallel for
        for (int j = 0; j < n; j++) {
            A[i][j] /= diag;
        }
        b[i] /= diag;

        // Прямой ход: обнуление элементов выше и ниже диагонали
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
    // Пример использования
    int n;
    cout << "Введите размерность системы: ";
    cin >> n;

    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);

    cout << "Введите коэффициенты матрицы A:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> A[i][j];
        }
    }

    cout << "Введите вектор b:\n";
    for (int i = 0; i < n; i++) {
        cin >> b[i];
    }

    // Решение системы
    gaussJordan(A, b);

    // Вывод результата
    cout << "Решение системы x:\n";
    for (int i = 0; i < n; i++) {
        cout << b[i] << " ";
    }
    cout << endl;

    return 0;
}
