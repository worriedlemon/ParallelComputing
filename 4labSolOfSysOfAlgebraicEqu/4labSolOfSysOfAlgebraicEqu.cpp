#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// ��������� ������
const int NUM_RUNS = 10; // ���������� ����������

// ������� ��� ������� ���� ������� ������-������� � ��������������� OpenMP
void gaussJordanParallel(vector<vector<double>>& A, vector<double>& b, int num_threads) {
    int n = A.size();

    // ������������� ���������� ������� ��� OpenMP
    omp_set_num_threads(num_threads);

    // ������ ��� ������ ������
    for (int k = 0; k < n; ++k) {
        // ����������� ������� ������� �� ������
#pragma omp parallel for
        for (int j = k + 1; j < n; ++j) {
            A[k][j] /= A[k][k];
        }
        b[k] /= A[k][k];
        A[k][k] = 1.0;

        // ������ ��� ���������� �� ������ �������
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            if (i != k) {
                double factor = A[i][k];
                for (int j = k + 1; j < n; ++j) {
                    A[i][j] -= factor * A[k][j];
                }
                b[i] -= factor * b[k];
                A[i][k] = 0.0;
            }
        }
    }
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
    cout << "Avg time: " << "\033[33m" << avg_time << "\033[0m" << " sec\r";
    cout.flush();  // ��������� �������
}

int main() {
    srand(time(0));  // ������������� ���������� ��������� �����
    int num_threads = 0;
    int n = -1;

    // ���� ���������� ��������� � ��������� ������ �� ��������� ��� ����
    while (true) {
        cout << "Enter number of threads (0 to exit): ";
        cin >> num_threads;
        if (num_threads == 0) {
            cout << "Program terminated.\n";
            return 0;
        }

        // ���� ����������� �������
        while (true)
        {
            cout << "Enter system size (0 to re-enter number of threads): ";
            cin >> n;
            if (n == 0) {
                break;  // ��������� ������ ����� ���������� ���������
            }

            // ��������� ��������� ������� ���������
            vector<vector<double>> A(n, vector<double>(n));
            vector<double> b(n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    A[i][j] = rand() % 100 + 1;  // ��������� ����� �� 1 �� 100
                }
                b[i] = rand() % 100 + 1;
            }

            // ��������� ������� ���������� � 100 ������������
            double total_time = 0.0;

            for (int i = 0; i < NUM_RUNS; i++) {
                // ������� ����� ������� A � ������� b ��� ������� ����������
                vector<vector<double>> A_copy = A;
                vector<double> b_copy = b;

                double start_time = omp_get_wtime();  // ����� ������
                gaussJordanParallel(A_copy, b_copy, num_threads);
                double end_time = omp_get_wtime();    // ����� ����������

                total_time += (end_time - start_time);  // ��������� �����

                // ����������� ���������
                double avg_time = total_time / (i + 1);  // ������� ����� ���������� �� ������� ������
                displayProgress(i + 1, NUM_RUNS, avg_time);
            }

            // ����� �������������� ����������
            std::cout << "\n\033[A\033[2K" << "Average execution time over " <<
                "\033[33m" << NUM_RUNS << "\033[0m"
                << " runs: " <<
                "\033[33m" << total_time / NUM_RUNS << "\033[0m"
                << " seconds." << std::endl;
        }
    }

    return 0;
}
