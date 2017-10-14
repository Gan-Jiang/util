'''
Different ways to sort a list.
'''
import random
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


def partition(arr, l, r):
    pivot = arr[l]
    left = l + 1
    right = r
    while left <= right:
        while left <= right and arr[left] <= pivot:
            left += 1

        while left <= right and arr[right] >= pivot:
            right -= 1
        if right < left:
            break
        arr[left], arr[right] = arr[right], arr[left]
    arr[l], arr[right] = arr[right], arr[l]
    return right


def quick_sort(data, first, last):
    #O(n log n), worst case O(n ** 2), in-place
    if first < last:
        splitpoint = partition(data, first, last)
        quick_sort(data, first, splitpoint - 1)
        quick_sort(data, splitpoint + 1, last)


def random_partition(arr, l, r):
    randindex = random.randint(l, r)
    arr[l], arr[randindex] = arr[randindex], arr[l]
    pivot = arr[l]
    left = l + 1
    right = r
    while left <= right:
        while left <= right and arr[left] <= pivot:
            left += 1

        while left <= right and arr[right] >= pivot:
            right -= 1
        if right < left:
            break
        arr[left], arr[right] = arr[right], arr[left]
    arr[l], arr[right] = arr[right], arr[l]
    return right


def random_quick_sort(data, first, last):
    if first < last:
        splitpoint = random_partition(data, first, last)
        quick_sort(data, first, splitpoint - 1)
        quick_sort(data, splitpoint + 1, last)


def counting_sort(array, maxval):
    #in-place counting sort, O(n + k)  non-comparative integer sorting algorithm, k is the range of the interger keys.
    m = maxval + 1
    count = [0] * m               # init with zeros
    for a in array:
        count[a] += 1             # count occurences
    i = 0
    for a in range(m):            # emit
        for c in range(count[a]): # - emit 'count[a]' copies of 'a'
            array[i] = a
            i += 1


def radix_counting_sort(data, exp):
    output = [0] * len(data)
    count = [0] * 10

    for i in data:
        index = i // exp
        count[index % 10] += 1

    for i in range(1, len(count)):
        count[i] += count[i - 1]

    i = len(data)-1
    while i>=0:
        index = (data[i] // exp)
        output[count[index % 10] - 1] = data[i]
        count[index % 10] -= 1
        i -= 1

    for i in range(len(data)):
        data[i] = output[i]


def radix_sort(data):
    #O(d * (n + b)), b is base, d is number of digits, d = log b (k), k is the maximum number. If k <= n**c, then O(n log b (n)), if b = n, then linear.
    max_val = max(data)
    exp = 1
    while max_val // exp > 0:
        radix_counting_sort(data, exp)
        exp *= 10


def bucket_sort(data, n):
    #O(n + k), k is the size of buckets, worst case will take O(n ** 2), used when the data is uniformly distributed.
    buckets = []
    for i in range(n):
        buckets.append([])
    min_val, max_val = min(data), max(data)
    for i in data:
        buckets[(n - 1) * (i - min_val) // (max_val - min_val)].append(i)

    res = []
    for i in buckets:
        insertion_sort(i)
        res += i
    return res