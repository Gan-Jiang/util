'''
Different ways to sort a list.
'''
def bubble_sort(data):
    #worst case: O(n ** 2), best case: O(n), average case: O(n ** 2), space O(1)
    if data == []:
        return []
    sort_index = len(data) - 1
    while sort_index != 0:
        n = 0
        for i in range(sort_index):
            if data[i] > data[i + 1]:
                data[i], data[i + 1] = data[i + 1], data[i]
                n = i
        sort_index = n


def selection_sort(data):
    #O(n ** 2) comparisons, O(n) swaps.
    for i in range(len(data)):
        # Find the minimum element in remaining
        # unsorted array
        min_idx = i
        for j in range(i + 1, len(data)):
            if data[min_idx] > data[j]:
                min_idx = j

        # Swap the found minimum element with
        # the first element
        data[i], data[min_idx] = data[min_idx], data[i]


def insertion_sort(data):
    #O(n**2), stable.
    # Traverse through 1 to len(arr)
    for i in range(1, len(data)):
        key = data[i]
        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i - 1
        while j >= 0 and key < data[j]:
            data[j + 1] = data[j]
            j -= 1
        data[j + 1] = key
    return data


def merge(arr, left_half, right_half):
    i = k = j = 0
    while i < len(left_half) and j < len(right_half):
        if left_half[i] <= right_half[j]:
            arr[k] = left_half[i]
            i += 1
        else:
            arr[k] = right_half[j]
            j += 1
        k += 1

    while i < len(left_half):
        arr[k] = left_half[i]
        i += 1
        k += 1

    while j < len(right_half):
        arr[k] = right_half[j]
        j += 1
        k += 1

def merge_sort(data):
    #O(n log n), space O(n)
    if len(data) > 1:
        m = len(data) // 2
        left_half = data[: m]
        right_half = data[m:]
        merge_sort(left_half)
        merge_sort(right_half)
        merge(data, left_half, right_half)