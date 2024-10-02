#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>  // ��� ������� rand() � srand()
#include <ctime>    // ��� ������� time()
#include <algorithm> // ��� std::min

// ������� ��� ���������� ������� �������
// ��������� ����� ������� �� start �� end
void insertionSort(std::vector<int>& arr, int start, int end) {
    for (int i = start + 1; i <= end; i++) {
        int key = arr[i];
        int j = i - 1;

        // �������� ��������, ����� �������� ���� � ������ �����
        while (j >= start && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// ������� ��� ������� ���� ��������������� ������
// ������� ����� ������� ����� ��������� left, mid � right
void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // ������� ��������� ������� ��� ������ � ������� �����
    std::vector<int> L(n1), R(n2);

    // �������� ������ � ��������� �������
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int i = 0; i < n2; i++)
        R[i] = arr[mid + 1 + i];

    // ���������� ��� ����� ������� � �������� ������
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // �������� ���������� ��������
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// ������� ��� ������������� ������� ������
// ����������� ������� �����, ���������� �� ������ � ��� ����
void mergeAllBlocks(std::vector<int>& arr, int num_threads, int block_size) {
    int n = arr.size();
    while (block_size < n) {
        // ������������ ���������� ������� ������
#pragma omp parallel for
        for (int i = 0; i < n; i += 2 * block_size) {
            int left = i;
            int mid = std::min(i + block_size - 1, n - 1);
            int right = std::min(i + 2 * block_size - 1, n - 1);

            // ���� ���� ��� �����, ������� ��
            if (mid < right) {
                merge(arr, left, mid, right);
            }
        }
        // ����������� ������ ������ ��� ���������� �������
        block_size *= 2;
    }
}

int main() {
    int num_threads;

    // ���� ���������� ������� ��� ������������� ����������
    std::cout << "Enter the number of threads: ";
    std::cin >> num_threads;

    // ������������� ���������� ������� ��� OpenMP
    omp_set_num_threads(num_threads);

    int n;

    // �������� ���� ���������
    while (true) {
        std::cout << "Enter array size (0 to change the number of threads): ";
        std::cin >> n;

        // ���� 0, �������� ���������� �������
        if (n == 0) {
            std::cout << "Enter new number of threads: ";
            std::cin >> num_threads;
            omp_set_num_threads(num_threads);
            continue;
        }

        const int num_tests = 100;  // ���������� ���������� ��� �����
        double total_time = 0.0;    // ����� ���� ������

        // ��������� ��������� ������ ��� �������� �������
        for (int test = 0; test < num_tests; test++) {
            srand(static_cast<unsigned>(time(0)));  // ��������� ���������� �������

            std::vector<int> arr(n);
            for (int i = 0; i < n; i++) {
                arr[i] = rand() % (n + 1);
            }

            // �������� ��������� �����
            double start_time = omp_get_wtime();

            // ���������� ������ ����� ��� ������� ������
            int block_size = (n + num_threads - 1) / num_threads;

            // ������������ ���������� ������ ������� �������
#pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                int start = thread_id * block_size;
                int end = std::min(start + block_size - 1, n - 1);

                // ��������� ������ ���� �����������
                if (start < n) {
                    insertionSort(arr, start, end);
                }
            }

            // ������������ ������� ��������������� ������
            mergeAllBlocks(arr, num_threads, block_size);

            // �������� �������� �����
            double end_time = omp_get_wtime();

            // ��������� ����� ����������
            total_time += (end_time - start_time);
        }

        // ��������� ������� ����� ����������
        double avg_time = total_time / num_tests;

        // ������� ���������
        std::cout << "Average sorting time for " <<
            "\033[33m" << n << "\033[0m"
            << " elements over " << num_tests << " tests: " <<
            "\033[33m" << avg_time << "\033[0m"
            << " seconds" << std::endl;
    }

    return 0;
}
