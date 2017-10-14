'''
Different ways to implement binary search.
Complexity O(log n)
'''
def binary_search_iterative(data, target):
    low, high = 0, len(data) - 1
    while low <= high:
        mid = low + (high - low) // 2
        if data[mid] == target:
            return True
        elif target > data[mid]:
            low = mid + 1
        else:
            high = mid - 1
    return False


def binary_search_recursive(data, target, low, high):
    if low > target:
        return False
    else:
        mid = low + (high - low) // 2
        if data[mid] == target:
            return True
        elif target > data[mid]:
            return binary_search_recursive(data, target, mid + 1, high)
        else:
            return binary_search_recursive(data, target, low, mid - 1)


def binary_search_need_check(data, target):
    low, high = 0, len(data) - 1
    while low + 1 < high:
        mid = low + (high - low) // 2
        if data[mid] == target:
            return True
        elif target > data[mid]:
            low = mid
        else:
            high = mid
    if data[low] == target or data[high] == target:
        return True
    return False

import bisect
def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

def find_lt(a, x):
    'Find rightmost value less than x'
    i = bisect.bisect_left(a, x)
    if i:
        return a[i-1]
    raise ValueError

def find_le(a, x):
    'Find rightmost value less than or equal to x'
    i = bisect.bisect_right(a, x)
    if i:
        return a[i-1]
    raise ValueError

def find_gt(a, x):
    'Find leftmost value greater than x'
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return a[i]
    raise ValueError

def find_ge(a, x):
    'Find leftmost item greater than or equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return a[i]
    raise ValueError

#bisect.insort_left(a, x, lo=0, hi=len(a))
#bisect.insort_right(a, x, lo=0, hi=len(a))