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

// ������� ��� ������� ���� ��������������� ������
void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // ������� ��������� ������� ��� �������� ���� �����������
    std::vector<int> L(n1), R(n2);

    // �������� ������ � ��������� �������
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int i = 0; i < n2; i++)
        R[i] = arr[mid + 1 + i];

    // ������� ��������� �������� ������� � arr[left..right]
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

    // �������� ���������� ��������, ���� ����
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

// ������� ��� ������� ���� ������ ����� ����������
void mergeBlocks(std::vector<int>& arr, int num_threads, int block_size) {
    int n = arr.size();

    // ������� ������ �� �������
    for (int i = 1; i < num_threads; i++) {
        int left = 0;
        int right = (i + 1) * block_size - 1;

        if (right >= n) {
            right = n - 1;
        }

        int mid = i * block_size - 1;

        // ������� ��� �������� �����
        merge(arr, left, mid, right);
    }
}

int main() {
    int num_threads;

    // ���� ����� ������� ���� ��� � ������
    std::cout << "Enter the number of threads: ";
    std::cin >> num_threads;

    // ��������� ����� �������
    omp_set_num_threads(num_threads);

    int n;

    while (true) {
        // ���� ������� �������
        std::cout << "Enter array size (0 to change the number of threads): ";
        std::cin >> n;

        if (n == 0) {
            // ��������� ����� �������
            std::cout << "Enter new number of threads: ";
            std::cin >> num_threads;
            omp_set_num_threads(num_threads);
            continue;  // ������� � ������ ����� ��� ������ ����� ������� �������
        }

        // �������������� ��������� ��������� �����
        srand(static_cast<unsigned>(time(0)));

        // ������� ������ � ��������� ��� ���������� ������� �� 0 �� n
        std::vector<int> arr(n);
        for (int i = 0; i < n; i++) {
            arr[i] = rand() % (n + 1);  // ���������� ��������� ����� �� 0 �� n
        }

        // ������ ��������� �������
        std::cout << "Unsorted array: ";
        for (int i = 0; i < n; i++) {
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;

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

        // ������� ��������������� ������
        mergeBlocks(arr, num_threads, block_size);

        double end_time = omp_get_wtime();

        // ������ ���������������� �������
        std::cout << "Sorted array: ";
        for (int i = 0; i < n; i++) {
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;

        // ����� ������� ���������� ����������
        std::cout << "Sorting time: " << (end_time - start_time) << " seconds" << std::endl;
    }

    return 0;
}
