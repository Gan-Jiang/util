'''
implement KMP algorithm to find the first occurrence of needle in haystack.
'''
def computeLps(needle, N):
    lps = [0] * N
    length = 0
    i = 1
    while i != N:
        if needle[length] == needle[i]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

def find_pattern(needle, haystack):
    if needle == '':
        return 0
    M = len(haystack)
    N = len(needle)

    lps = computeLps(needle, N)

    i = j = 0
    while i != M:
        if haystack[i] == needle[j]:
            i += 1
            j += 1
            if j == N:
                return i - N
        else:
            if j > 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1