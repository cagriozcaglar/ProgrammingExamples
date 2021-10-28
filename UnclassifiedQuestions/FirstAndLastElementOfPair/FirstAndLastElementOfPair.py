"""
cons(a, b) constructs a pair, and car(pair) and cdr(pair) returns the first and last element of that pair.
For example, car(cons(3, 4)) returns 3, and cdr(cons(3, 4)) returns 4.

Given this implementation of cons:

def cons(a, b):
    def pair(f):
        return f(a, b)
    return pair

Implement car and cdr.

Solutions:
1) https://galaiko.rocks/posts/dcp/problem-5/
2) https://tech-cookbook.com/2018/11/09/daily-coding-problem-consa-b-constructs-a-pair-and-carpair-and-cdrpair-returns-the-first-and-last-element-of-that-pair/
3) https://stackoverflow.com/questions/52481607/dont-understand-the-inner-function-in-python
"""


def cons(a, b):
    def pair(f):
        return f(a, b)
    return pair


def car(pair):
    def returnFirst(a, b):
        return a
    return pair(returnFirst)


def cdr(pair):
    def returnSecond(a, b):
        return b
    return pair(returnSecond)


# Create a pair
pair_3_4 = cons(3, 4)

print(f"pair_3_4: {pair_3_4}")
# pair_3_4: <function cons.<locals>.pair at 0x1091444c0>
print(f"type(pair_3_4): {type(pair_3_4)}")
# type(pair_3_4): <class 'function'>
print(f"pair_3_4.__closure__[0].cell_contents: {pair_3_4.__closure__[0].cell_contents}")
# pair_3_4.__closure__[0].cell_contents: 3
print(f"pair_3_4.__closure__[1].cell_contents: {pair_3_4.__closure__[1].cell_contents}")
# pair_3_4.__closure__[1].cell_contents: 4
print(f"car(cons(3, 4)): {car(cons(3, 4))}")
# car(cons(3, 4)): 3
print(f"cdr(cons(3, 4)): {cdr(cons(3, 4))}")
# cdr(cons(3, 4)): 4
pair_3_4(print)
# 3 4
