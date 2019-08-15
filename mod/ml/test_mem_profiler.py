import itertools


@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    c = a + b
    d = itertools.chain(a, b)
    e = set()
    for i in d:
        e.add(i)
    del b
    return a


if __name__ == "__main__":
    my_func()
