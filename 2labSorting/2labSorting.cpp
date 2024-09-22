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

    int n = 1;

    while (n != 0) {
        // �������������� ��������� ��������� �����
        srand(static_cast<unsigned>(time(0)));

        // ������� ������ � ��������� ��� ���������� ������� �� 0 �� n
        std::vector<int> arr(n);
        for (int i = 0; i < n; i++) {
            arr[i] = rand() % (n + 1);  // ���������� ��������� ����� �� 0 �� n
        }

        // ������ ��������� �������
        /*std::cout << "source array: ";
        for (int i = 0; i < arr.size(); i++)
            std::cout << arr[i] << " ";
        std::cout << std::endl;*/

        clock_t tStart = clock(); // ������� ������� ����������

        // ��������� ����������
        insertionSort(arr);

        // ������ ���������������� �������
        /*std::cout << "sorted array: ";
        for (int i = 0; i < arr.size(); i++)
            std::cout << arr[i] << " ";
        std::cout << std::endl;*/

        // ����� �������
        printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

        // ���� ������� �������
        std::cout << "enter array size: ";
        std::cin >> n;
    }

    return 0;
}