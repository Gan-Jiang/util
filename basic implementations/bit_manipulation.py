'''
Some basic bit operations.
'''
def set_bit(n, i):
    '''
    set / turn on the ith bit
    :param n:
    :param i:
    :return:
    '''
    return n | (1 << i)

def check_bit(n, i):
    '''
    check if the ith bit is on
    :param n: the number we want to check
    :param i: the bit
    :return:
    '''
    return n & (1 << i)

def multiply_2(n):
    #divided by 2 n >> 1
    return n << 1

def toggle_bit(n, i):
    '''
    flip the status of i-th bit.
    :param n:
    :param i:
    :return:
    '''
    return n ^ (1 << i)

def clear_bit(n, j):
    '''
    clear / turn off the j-th bit
    :param n:
    :param j:
    :return:
    '''
    return n & (~ (1 << j))

def least_significant_bit(n):
    '''
    get the value of the least significant bit that is on (first from the right)
    :param n:
    :return:
    '''
    return n & (-n)

def turn_on_all_bits_m(n, m):
    '''
    turn on all bits in a set of size m
    :param n: number
    :param m: size
    :return:
    '''
    return n | (1 << m - 1)

def get_remainder(n, m):
    return n & (m - 1)

def is_power_2(n):
    return (n & (n - 1)) == 0

def turn_off_last_bit(n):
    return n & (n - 1)

def turn_on_last_zero(n):
    return n | (n + 1)

def turn_off_last_consecutive_ones(n):
    return n & (n + 1)

def turn_on_last_consecutive_zeros(n):
    return n | (n - 1)

