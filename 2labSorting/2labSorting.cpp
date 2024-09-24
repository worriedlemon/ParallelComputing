#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>  // ��� ������� rand() � srand()
#include <ctime>    // ��� ������� time()

// ������� ��� ���������� ������� �������
void insertionSort(std::vector<int>& arr) {
    int n = arr.size();

    // �������������� � ������� OpenMP
#pragma omp parallel for shared(arr)
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;

        // �������� �������� arr[0..i-1], ������� ������ �����,
        // �� ���� ������� ������ �� �� ������� �������
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

int main() {
    int num_threads;

    // ���� ����� ������� ���� ��� � ������
    std::cout << "Enter threads count: ";
    std::cin >> num_threads;

    // ��������� ����� �������
    omp_set_num_threads(num_threads);

    int n;

    while (true) {
        // ���� ������� �������
        std::cout << "Enter array size (0 for edit threads count): ";
        std::cin >> n;

        if (n == 0) {
            // ��������� ����� �������
            std::cout << "Enter threads count: ";
            std::cin >> num_threads;
            omp_set_num_threads(num_threads);
            continue;  // ������� � ������ ����� ��� ������ ����� ������� �������
        }

        double total_time = 0;
        int test_count = 100;

        // ���������� 100 ������
        for (int test = 0; test < test_count; test++) {
            // �������������� ��������� ��������� �����
            srand(static_cast<unsigned>(time(0)));

            // ������� ������ � ��������� ��� ���������� ������� �� 0 �� n
            std::vector<int> arr(n);
            for (int i = 0; i < n; i++) {
                arr[i] = rand() % (n + 1);  // ���������� ��������� ����� �� 0 �� n
            }

            // ������� ������� ���������� ���������� � �������������� OpenMP
            double start_time = omp_get_wtime();

            // ��������� ����������
            insertionSort(arr);

            double end_time = omp_get_wtime();

            // ��������� ����� ����������
            total_time += (end_time - start_time);
        }

        // ���������� �������� ������� ����������
        double avg_time = total_time / (float)test_count;

        // ����� �������� ������� ���������� ����������
        std::cout << "Average time taken " << n << " elements: " << avg_time << " seconds" << std::endl;
    }

    return 0;
}
