#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>  // ��� ������� rand() � srand()
#include <ctime>    // ��� ������� time()

// ������� ��� ���������� ������� �������
void insertionSort(std::vector<int>& arr, int start, int end) {
    for (int i = start + 1; i <= end; i++) {
        int key = arr[i];
        int j = i - 1;

        while (j >= start && arr[j] > key) {
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

            // ������ �����
            int block_size = n / num_threads;

            // ������������ ���������� ������
#pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                int start = thread_id * block_size;
                int end = (thread_id == num_threads - 1) ? n - 1 : (start + block_size - 1);

                insertionSort(arr, start, end);
            }

            // �� ������ ����� ����� ������������ ��������� ������� ������ ��� ������ ���������� �������

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
