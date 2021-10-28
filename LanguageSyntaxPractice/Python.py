from collections import defaultdict
from collections import deque

# for / else clause: https://book.pythontips.com/en/latest/for_-_else.html

# SORTING
# In-place sorting using object.sort() vs. out-of-place sorting using sorted()
values = [ [5,1], [4,1], [3,1], [8,1], [7,1] ]
values.sort(key = lambda x: x[0])
values
# [[3, 1], [4, 1], [5, 1], [7, 1], [8, 1]]
values = [ [5,1], [4,1], [3,1], [8,1], [7,1] ]
sortedValues = sorted(values, key = lambda x: x[0])
sortedValues
# [[3, 1], [4, 1], [5, 1], [7, 1], [8, 1]]
values
# [[5, 1], [4, 1], [3, 1], [8, 1], [7, 1]]

# Example 1: Sort objects by field
from typing import List
class Box:
    def __init__(self, height: int, width: int, depth: int):
        self.height = height
        self.width = width
        self.depth = depth
    # Serialize the box and print
    def toString(self):
        print(f"({self.height}, {self.width}, {self.depth})")

def printBoxList(boxes: List[Box]):
    [box.toString() for box in boxes]

boxes : List[Box] = [
    Box(3, 4, 5),
    Box(5, 4, 3),
    Box(2, 1, 0)
]
printBoxList(boxes)
'''
(3, 4, 5)
(5, 4, 3)
(2, 1, 0)
'''
# Sort by height, default is ascending / increasing order
boxes.sort(key= lambda box : box.height)
printBoxList(boxes)
'''
(2, 1, 0)
(3, 4, 5)
(5, 4, 3)
'''
# Sort by height in reverse order, which is decreasing / descending order
boxes.sort(key= lambda box : box.height, reverse=True)
'''
(5, 4, 3)
(3, 4, 5)
(2, 1, 0)
'''

# Mapping: Map a function to a variable
# https://www.programiz.com/python-programming/methods/built-in/map
routes = [ [1, 2, 7], [3, 6, 7] ]
routes2 = map(set, routes)
list(routes2)
# [{1, 2, 7}, {3, 6, 7}]
def calculateSquare(n):
    return n*n
numbers = (1, 2, 3, 4)
result = map(calculateSquare, numbers)
list(result)
# [1, 4, 9, 16]

# Ranges
range(5)      # [0,1,2,3,4]
range(1,5)    # [1,2,3,4]
range(1,5,2)  # [1,3]  (3rd argument of range() method is the stepSize=2)

# Numeric values of booleans
int(True)
# 1
int(False)
# 0

# Random numbers
import random
# Returns number between [0,1) (left inclusive, right exclusive)
random.random()
# 0.052613705472609906
# Pick an integer between [0, n) ( == [0, n-1])
n = 5
pick = int(random.random() * n)
# Returns integers between 0-4.
# Returns an integer between [a,b] (both inclusive)
random.randint(0, 10)
# E.g. select a random pivot index between [left, right] (both inclusive)
left = 3
right = 5
pivot_idx = random.randint(left, right)
# Choose an element from a list
import random
random.choice([1,2,3])
# Returns one of 1,2,3

# One line swap
a = 3
b = 4
a, b = b, a
a
# 4
b
# 3

# Math functions
import math
# Ceiling and floor of a decimal
x = 4.3
math.ceil(x)
# 5
math.floor(x)
# 4
# isclose(): Return True if the values a and b are close to each other and False otherwise.
# math.isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0)
math.isclose(x, 4.30000000000000001)
# True
# Check if a number is nan
x = math.nan  # do not use None here
math.isnan(x)
# True
# Product of numbers in an iterable
x = [1,2,3,4]
math.prod(x)
# 24
# Return the Real value x truncated to an Integral (usually an integer).
x = 4.3
math.trunc(x)
4
# exp(): Return e raised to the power x, where e = 2.718281… is the base of natural logarithms.
math.exp(x)
# 73.69979369959579
# log(x): math.log(x[, base])
# With one argument, return the natural logarithm of x (to base e).
# With two arguments, return the logarithm of x to the given base, calculated as log(x)/log(base).
math.log(x)
# 1.4586150226995167
# log2()
math.log2(x)
# 2.1043366598147357
# log10()
math.log10(x)
# 0.6334684555795865
# pow(x,y): Return x raised to the power y.
y = 1
math.pow(x, y)
# 4.3
# sqrt(x): Return the square root of x.
math.sqrt(x)
# 2.073644135332772
# Return the Euclidean distance between two points p and q, each given as a sequence (or iterable) of coordinates.
# The two points must have the same dimension.
x = [1.0, 1.0]
y = [4.0, 5.0]
math.dist(x,y)
5.0  # 3^2 + 4^2 = 5^2
# .pi: The mathematical constant π = 3.141592
math.pi
# 3.141592653589793
# .e: The mathematical constant e = 2.718281
math.e
# 2.718281828459045
# A floating-point positive infinity. (For negative infinity, use -math.inf.) Equivalent to the output of float('inf').
math.inf
# inf
# A floating-point “not a number” (NaN) value. Equivalent to the output of float('nan').
math.nan
# nan
# math.gcd(*integers): Return the greatest common divisor of the specified integer arguments.
math.gcd(10, 25)
# 5
# math.lcm(*integers): Return the least common multiple of the specified integer arguments.
math.lcm(10, 25)
# 50

# isinstance() check: returns True if the specified object is of the specified type, otherwise False.
# https://www.w3schools.com/python/ref_func_isinstance.asp
x = isinstance(5, int)
x
# True

# Enumeration in loops
aList = [1, 2, 3]
for index, value in enumerate(aList):
    print(f"aList[{index}]: {aList[index]}")
'''
aList[0]: 1
aList[1]: 2
aList[2]: 3
'''

# Loops
# Iterate over pairs by careful off-by-one
values = [0, 1, 2, 3]
for i in range(0, len(values)-1):
    print(f"({values[i]}, {values[i+1]})")
'''
(0, 1)
(1, 2)
(2, 3)
'''
# Reverse ranges using reversed() method
list(reversed(range(5)))
# [4, 3, 2, 1, 0]
# Use product() method to generate cartesian product, to iterate over nested loops
from itertools import product
list(product(range(2),range(3)))
# [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
list(product([1,2], [3,4,5]))
# [(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
nrows = 2
ncols = 3
for row, col in list(product(range(nrows),range(ncols))):
    print(f"({row}, {col})")
'''
(0, 0)
(0, 1)
(0, 2)
(1, 0)
(1, 1)
(1, 2)
'''
# Iterate over character pairs of two strings to find longest common prefix
str1 = "abcd"
str2 = "abc"
substring = ""
for ch1, ch2 in zip(str1, str2):
    if ch1 != ch2:
        print(f"{substring}:, {len(substring)}")
    substring = substring + ch1
print(f"{substring}: {len(substring)}")
# abc: 3

# Division: Integer vs. Floating point. Be careful
# Single divide: / => floating point division
(1+2) / 2
# 1.5
# Double divide: // => integer division
(1+2) // 2
# 1
# E.g. in binary search: mid = (low+high) // 2 (double divide, not single divide)


# GENERATORS
# https://docs.python.org/3/tutorial/classes.html#generators
# Example 1: Range generator
def range_generator(a, b):
    current = a
    while current < b:
        # This yield "returns" a value from the function and pauses it
        yield current
        # Once the function is "woken up" by another call to next(...), it will resume
        # by continuing with the next statement (current += 1), until it either
        # hits another yield or reaches the end of the function
        current += 1
    # When we get here, the generator is finished.
# Create a new range_generator object for the numbers 10 to 20.
ten_to_twenty_generator = range_generator(10, 20)
# Get the first 3 values out of the range_generator object we made.
print(next(ten_to_twenty_generator))
# 10
print(next(ten_to_twenty_generator))
# 11
print(next(ten_to_twenty_generator))
# 12
# Example 2: Reverse an array
def reverse(data):
    for index in range(len(data)-1, -1, -1):
        yield data[index]
for char in reverse('golf'):
    print(char)
# f
# l
# o
# g

# TODO
#   MULTI-THREADING / CONCURRENCY
#   https://realpython.com/intro-to-python-threading/

# Binary Search using bisect (=bisect_right), bisect_left
# https://docs.python.org/3/library/bisect.html
# Given array a, number x, low index lo, high index hi, call:
# bisect.[bisect|bisect_right|bisect_left](a, x, lo, hi)
#  1) bisect (= bisect_right) returns the index of insertion point which comes after (to the right of)
#  any existing entries of x in a.
#  2) bisect_left returns the index of insertion point which comes before (to the left of)
#  any existing entries of x in a.
import bisect
a = [1, 2, 3, 3, 4, 5]
#Id: 0  1  2  3  4  5
bisect.bisect(a, 3)
# 4
bisect.bisect_right(a, 3)
# 4
bisect.bisect_left(a, 3)
# 2
# When element does not exist, bisect() returns the first index higher than the value (returns index to the right)
bisect.bisect(a, 2.5)
# 2

# Line break using backslash "\"
a = 1 + 2 + \
    3 + 4
a
# 10

# Maximum / minimum integers
import sys
sys.maxint
# 9223372036854775807
-sys.maxint - 1
# -9223372036854775808
# Numbers higher / lower than any others: Float
float('inf')
# inf
# Check:
10 ** (10) < float('inf')
# True
float('-inf')
# -inf
# Check:
-10 ** (11) > float('-inf')
# True

# Easy array initialization
populateArrayEasily = [True] * 10

# Initialize a list of lists
listOfLists = [[]]
# Concatenate lists
val = ["a"] + ["b"]  # Result: ['a', 'b']

# ord(): Convert character to integer representing Unicode character
# https://www.programiz.com/python-programming/methods/built-in/ord
ord("a")  # 97
# chr(): Converts an integer representing unicode code point of the character, to the character (a string)
# https://www.programiz.com/python-programming/methods/built-in/chr
chr(97)   # 'a'


# bin(): Convert integer to binary string
# https://www.programiz.com/python-programming/methods/built-in/bin
bin(12)  # '0b1100'

# map function for converting types
# https://www.geeksforgeeks.org/python-program-to-convert-list-of-integer-to-list-of-string/
x = [1,2,3]
map(str, x)  # ['1', '2', '3']

# Enums
from enum import Enum
class Weekday(Enum):
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6
    SUNDAY = 7
print(Weekday.WEDNESDAY.name)   # "WEDNESDAY"
print(Weekday.WEDNESDAY.value)  # 3

# Swap values easily
a = 5
b = 4
a, b = b, a
print(a)
# 4
print(b)
# 5

# Generate frequency map of a container using Counter
from collections import Counter
nums = [1,1,2,3,3,3]
Counter(nums)
# Counter({3: 3, 1: 2, 2: 1})
# Now, convert to dictionary
dict(Counter(nums))
# {1: 2, 2: 1, 3: 3}
# Get most common elements using Counter.most_common() method:
# https://docs.python.org/3/library/collections.html#collections.Counter.most_common
Counter('abracadabra').most_common(3)
# [('a', 5), ('b', 2), ('r', 2)]

# Hash functions
# sha256
# E.g. you can use to hash trees as in Merkle trees
from hashlib import sha256
S = sha256()
S
# <sha256 _hashlib.HASH object @ 0x1038c5630>
S.update(1)
S.hexdigest()
# 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'

# LOGICAL OPERATORS / KEYWORDS
# all()
seen = [False] * 10
seen[0] = True
all(seen)  # Returns false
# and / or / not ( Forget &&, ||, !, this is not Java)

# TERNARY OPERATOR
# Python does not have ternary operator "? :".
# Instead "[var1] if [condition] else [var2]" is used as the ternary operator
a = 3
b = 4
condition = True
x = a if condition else b
x
# 3

# EXCEPTIONS / ERRORS
# https://docs.python.org/3/tutorial/errors.html
try:
    10 * (1/0)
except ZeroDivisionError:  # When exception of this type is caught
    print("Exception handled here: ZeroDivisionError")
except (RuntimeError, TypeError, NameError):  # Can have multiple exception types
    print("Do not handle these exceptions")
    pass
'''
Exception handled here: ZeroDivisionError
'''
# else clause after try: try-except-else caluse: Else is executed if no exception is caught in try block
try:
    10 * 5
except ZeroDivisionError:  # When exception of this type is caught
    print("Exception handled here: ZeroDivisionError")
else:
    print("No exceptions: All good")
'''
50
No exceptions: All good
'''
# The raise() statement allows the programmer to force a specified exception to occur.
try:
    a, b = 10, 0
    if b == 0:
        raise ZeroDivisionError
    else:
        a / b
except ZeroDivisionError:  # When exception of this type is caught
    print("Exception handled here: ZeroDivisionError")
'''
Exception handled here: ZeroDivisionError
'''
# finally clause: Optional in try-[except]-[finally] clause.
# The finally clause will execute as the last task before the try statement completes.
# The finally clause runs whether or not the try statement produces an exception.
try:
    10 / 0
except ZeroDivisionError:
    print("Exception handled here: ZeroDivisionError")
finally:
    print("In finally clause: This is printed no matter whether an exception is caught ot not")
'''
Exception handled here: ZeroDivisionError
In finally clause: This is printed no matter whether an exception is caught ot not
'''
# Summary of try-except-else-finally clause with a method
def divide(x, y):
    try:
        result = x / y
    except ZeroDivisionError:
        # Executed only if ZeroDivisionError exception is caught in try block
        print("division by zero!")
    else:
        # Executed only if no exception is caught in try block
        print("result is", result)
    finally:
        # Executed always, regardless of whether an exception is caught in try block or not
        print("executing finally clause")
divide(2, 1)
'''
result is 2.0
executing finally clause
'''
divide(2, 0)
'''
division by zero!
executing finally clause
'''
divide("2", "1")
'''
executing finally clause
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
File "<stdin>", line 3, in divide
TypeError: unsupported operand type(s) for /: 'str' and 'str'
'''

# TESTING WITH PYTEST
import pytest
def getVal(x: int):
    return x+1
# Test starts here
def test_capital_case():
    print("testing this one now")
    assert getVal(5) == 6
    print("Test is correct if it reached here")
test_capital_case()
'''
testing this one now
Test is correct if it reached here
'''

######################
# DATA STRUCTURES
######################
# STRINGS
# Substrings of strings
text = "abcde"
print(text[0])
# 'a'
print(text[1:])
# 'bcde'
# Take the index of first occurrence
# The index() method finds the first occurrence of the specified value.
# https://www.w3schools.com/python/ref_string_index.asp
txt = "Hello"
x = txt.index("el")
print(x)
# 1
# isSubstring check
"cde" in "abcdef"
# True
"cde" in "abcdf"
# False
# Sort characters of a string
sorted("cdea")
# ['a', 'c', 'd', 'e']
"".join(sorted("cdea"))
# 'acde'
# Iterate over characters of a string
for ch in "abc":
    print(ch)
# a
# b
# c
# Sort a string
a_string = "cba"
sorted_characters = sorted(a_string)
a_string = "".join(sorted_characters)
print(a_string)
# abc
# Split string by a list of delimiters
import re
a='Beautiful, is; better*than\nugly'
re.split('; |, |\*|\n',a)
# ['Beautiful', 'is', 'better', 'than', 'ugly']



# Character count map of a string
import collections
dict(collections.Counter("abcde"))
# {'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1}
# Convert string to character array
# Methods: https://www.delftstack.com/howto/python/split-string-to-a-list-of-characters/
# 1. Cast to list
string = "abc"
list(string)
# ['a', 'b', 'c']
# 2. List comprehension
lst = [x for x in string]
lst
# ['a', 'b', 'c']
# Easy serialized expression parsing trick
expression = "((a+b)*c)/d"
# Replace '(' -> '( ' and ')', ' )', to tokenize easier
expression.replace('(', ' ( '). replace(')', ' ) ')
# ' (  ( a+b ) *c ) /d'
expression.replace('(', ' ( '). replace(')', ' ) ').split()
# ['(', '(', 'a+b', ')', '*c', ')', '/d']
# Methods of interest: isalpha(), isdecimal(), isdigit(), isnumeric()
"abcde".isalpha()
# True
"1".isdigit()
# True
"123".isdigit()
# True
"123".isnumeric()
# True
"234".isdecimal()
# True
# Methods of interest: startswith((), endswith(), isdigit(), isnumeric()
"abc".startswith("a")
# True
"abc".endswith("c")
# True
"abc".islower()
# True
"ABC".isupper()
# True
# Iterate over words and their characters
words = ["be", "se", "on"]
for word in words:
    for letter in word:
        print(f"do something here: {letter}")
'''
do something here: b
do something here: e
do something here: s
do something here: e
do something here: o
do something here: n
'''
# Cartesian product of characters of two strings using product() method
from itertools import product
a = "123"
b = "456"
product(a,b)
# <itertools.product object at 0x10803bc80>
list(product(a,b))
# [('1', '4'), ('1', '5'), ('1', '6'), ('2', '4'), ('2', '5'), ('2', '6'), ('3', '4'), ('3', '5'), ('3', '6')]

# LISTS
# 1. Slicing:
a = [0, 1, 2, 3, 4]
a[2:4]    # left index (2) inclusive, right index (4) exclusive
# [2, 3]
# 2. Append vs. Extend vs. +=
# Append: Appends object at the end
x = [1, 2, 3]
x.append([4, 5])
print(x)   # [1, 2, 3, [4, 5]]
# Extend: Extends list by appending elements from the iterable.
# CAREFUL: Grow a list with another list using extend(), not append()
x = [1, 2, 3]
x.extend([4, 5])
print(x)   # [1, 2, 3, 4, 5]
# += to append element to a list
a = [1,2,3]
a += [4]
a
# [1, 2, 3, 4]
# 3. Remove element from lists
l = [1,2,3,4,5]
# Remove element at index 0 using pop(index)
l.pop(0)
# 1
l
# [2, 3, 4, 5]
# Remove last element using pop()
l.pop()
# 5
l
# [2, 3, 4]
# Insert to the beginning ( .insert(0,x) ) and end of a list ( .append(x) )
a = [1,2,3,4,5]
a
# [1, 2, 3, 4, 5]
a.insert(0, 100)
a
# [100, 1, 2, 3, 4, 5]
a.append(200)
a
# [100, 1, 2, 3, 4, 5, 200]
# Remove element at the beginning ( .pop(0) ) and end of a list ( .pop(-1) )
a = [100, 1, 2, 3, 4, 5]
a
# [100, 1, 2, 3, 4, 5]
# Remove first element
a.pop(0)
# 100
a
# [1, 2, 3, 4, 5]
# Remove last element
a.pop(-1)
# 5
a
# [1, 2, 3, 4]
# Mean / median / mode / stdev / variance of a list (use statistics library)
from statistics import mean, median, mode, stdev, variance
theList = [1,2,3,4]
mean(theList)
# 2.5
median(theList)
# 2.5
mode(theList)
# 1
stdev(theList)
# 1.2909944487358056
variance(theList)
# 1.6666666666666667
# Return combinations of size k
# https://docs.python.org/3/library/itertools.html#itertools.combinations
from itertools
list(itertools.combinations([1,2,3,4], 2))
# [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
list(itertools.combinations("abcd", 2))
# [('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')]
# 1. Serialize / join list into string using join() method
",".join(["1","2"])
# '1,2'
# Note: Passing ints in the list will return failure: ",".join([1,2]) => error
# 2. Deserialize string into list using split() method
'1,2'.split(",")
# ['1', '2']

# Add an element to each list in a list of lists
y = [[1, 2, 1, 1], [3, 4, 1], [5, 6, 1]]
# >>> y
# [[1, 2, 1, 1], [3, 4, 1], [5, 6, 1]]
[yy.append("AA") for yy in y]
# [None, None, None]
# >>> y
# [[1, 2, 1, 1, 'AA'], [3, 4, 1, 'AA'], [5, 6, 1, 'AA']]
# Extended Slicing in Lists: https://docs.python.org/release/2.3.5/whatsnew/section-slices.html
L = range(10)
# return elements with index multiple of 2
L[::2]
# [0, 2, 4, 6, 8]
# Reverse the list (nothing for start, nothing for end, -1 for stepping backwards from end to start)
L[::-1]
# [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
# Also works for tuples, arrays, and strings
s = 'abcd'
s[::2]
# 'ac'
s[::-1]
# 'dcba'


# ARRAYS
# Cumulative sums of array elements
# Method 1: Cumulative sum using accumulate (easier)
from itertools import accumulate
# Calculate cumulative sum of the list, return a generator (not a list)
accumulate([1, 2, 3, 4, 5])
# <itertools.accumulate object at 0x10ebec980>
# Convert generator to list
list(accumulate([1, 2, 3, 4, 5]))
# [1, 3, 6, 10, 15]
# Read from generator
values = accumulate([1,2,3,4,5])
for value in values:
    print(value)
'''
1
3
6
10
15
'''
# Method 2: Cumulative sum of an array using slicing
a = [1, 2, 3, 4, 5]
k = 3
sum(a[0:k:1])  # Cumulative cum of first k elements
# 6
cumsum_a = [sum(a[0:x:1]) for x in range(0, len(a)+1)]
cumsum_a
# [0, 1, 3, 6, 10, 15]
cumsum_a[1:]
# [1, 3, 6, 10, 15]

# 2-D ARRAYS
# Create 2-D matrix of 1's of size m x n
n = 10
m = 20
twoDimArray = [[1] * n for _ in range(m)]
# 10x10 2-D matrix with all dots in cells
empty_board = [["."] * n for _ in range(n)]
# Find the max of 2-D array
maxVal = max(max(x) for x in twoDimArray)

# Reverse a list: https://www.programiz.com/python-programming/methods/list/reverse
a = [1,2,3,4,5]
#   In-place reverse
a.reverse()  # a will have the reverse values afterwards
#   Slicing (not in-place)
b = a[::-1]  # Reversal is not in-place: a will have the original list, b will have the reverse list
#   Reversed keyword, followed by casting as a list (not in-place)
b = list(reversed(a))   # reversed() returns iterator, casting to list returns the reversed list
                        # Not in-place: a will have the original list, b will have the reverse list

# Iterate over matrix cells in 4-potential directions:
# explore the 4 potential directions around
row, col = 0, 0
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
for ro, co in directions:
    next_row, next_col = row + ro, col + co


# MAP / HASHMAP / DICTIONARY
# Get element from map with default value
theMap = {"a" : 1, "b" : 2}
key = "c"
theMap.get(key, 0)
# E.g. increment value of b
theMap["b"] = theMap.get("b", 0) + 1
# Add new elements to dictionary
a = {1:2, 3:4}
a.update({5:6})
# {1: 2, 3: 4, 5: 6}
# Delete an element from a map using del keyword
a = dict({1: '1', 2: '2', 3: '3'})
del a[2]
a
# Delete an element from map using pop(item) method
a = {1:2, 3:4, 5:6}
a.pop(1)
# 2
a
# {3: 4, 5: 6}
a.pop(100)
'''
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
KeyError: 100
'''
# Instead, safe pop() using default boolean argument
a.pop(200, False)
# False
# Check equality of dictionaries using == operator
a = {1:2, 3:4}
b = {3:4, 1:2}
a == b
# True
# Note: even if order of elements is different, only the content without order is compared
# {1: '1', 3: '3'}
# Initialize map from list of lists
b = [ [1, 2], [3, 4] ]
c = {items[0] : items[1] for items in b}
c
# {1: 2, 3: 4}
# Init example:
numberMap = {i : True for i in range(5)}
numberMap
# {0: True, 1: True, 2: True, 3: True, 4: True}

# 2-level Map
from collections import defaultdict
twoLevelMap = defaultdict(lambda: defaultdict(int))  # HashMap<HashMap<Integer>>
twoLevelMap[1][2] = 'a'
twoLevelMap
# defaultdict(<function <lambda> at 0x1080e9b80>, {1: defaultdict(<class 'int'>, {2: 'a'})})


# TREEMAP
# Roundabout way is to sort yourself
sd = {'c': 3, 'a': 1, 'b': 2}
print(sd)
# {'c': 3, 'a': 1, 'b': 2}
# Sort by key and return new dict
dict(sorted(sd.items(), key=lambda item: item[0]))
# {'a': 1, 'b': 2, 'c': 3}
# Sort by value (in reverse order, hence decreasing order) and return new dict
dict(sorted(sd.items(), key=lambda item: item[1], reverse=True))
# {'c': 3, 'b': 2, 'a': 1}
# Another way: use SortedDict
# See it here: http://www.grantjenks.com/docs/sortedcontainers/
from sortedcontainers import SortedDict
sd = SortedDict({'c': 3, 'a': 1, 'b': 2})
print(sd)
# SortedDict({'a': 1, 'b': 2, 'c': 3})
sd.popitem(index=-1)
# ('c', 3)
# Use OrderedDict: dict subclass that remembers the order entries were added
# https://docs.python.org/3/library/collections.html#collections.OrderedDict
import collections
d = {2: 3, 1: 89, 4: 5, 3: 0}
orderedD = collections.OrderedDict(sorted(d.items()))
# More readable version of the line above
orderedD = collections.OrderedDict(sorted(d.items(), key= lambda x: x[0]))
orderedD
# OrderedDict([(1, 89), (2, 3), (3, 0), (4, 5)])
# When you add new elements, the insertion order is preserved, and the element is added to end
orderedD[-100] = "-100"
orderedD
# OrderedDict([(1, 89), (2, 3), (3, 0), (4, 5), (-100, '-100')])
# When you add an existing element, insertion order is preserved, position is kept the same for the existing element
orderedD[1] = "1"
orderedD
# OrderedDict([(1, '1'), (2, 3), (3, 0), (4, 5), (-100, '-100')])
# use move_to_end() method to move an element to end of the keys list (Good use case for implementing LRU cache)
orderedD.move_to_end(1)
orderedD
# OrderedDict([(2, 3), (3, 0), (4, 5), (-100, '-100'), (1, '1')])
# Use popitem() method to remove a key
orderedD.popitem(1)
# (1, '1')
orderedD
# OrderedDict([(2, 3), (3, 0), (4, 5), (-100, '-100')])


# SETS
# Initialize
a = set()
# Add / Remove
a.add("a")
print(a)
# set(['a'])
a.remove("a")
print(a)
# set([])
# Remove element not in set, returns key error
a.remove("b")
# ## Traceback (most recent call last):
# ## File "<stdin>", line 1, in <module>
# ## KeyError: 'b'
# In such cases where key does not exist in set, use discard() method instead of remove()
a.discard("b")
print(a)
set([])
# Check set equality using == operator
a = set([1,2,3])
b = set([3,2,1])
a == b
# True
# Convert a list to set
theList = [1,2,3,4,3,2,1]
theSet = set(theList)
theSet
# {1, 2, 3, 4}

# TREESET
# In Python, it is called SortedSet
# See it here: http://www.grantjenks.com/docs/sortedcontainers/
from sortedcontainers import SortedSet
ss = SortedSet('abracadabra')
print(ss)
# SortedSet(['a', 'b', 'c', 'd', 'r'])
ss.bisect_left('c')
# 2

# STACK
# Stack Implementation using lists

stackUsingList = []
stackUsingList.append("a")  # push()
stackUsingList.append("b")  # push()
stackUsingList
# ['a', 'b']
stackUsingList.pop()        # pop()
# 'b'
stackUsingList
# ['a']
# Peek(): Get top element using slicing
stackUsingList[-1]
# 'a'
# Stack Implementation using deque
from collections import deque
stackUsingDeque = deque()
stackUsingDeque.append("a")  # push()
stackUsingDeque.append("b")  # push()
stackUsingDeque.append("c")  # push()
stackUsingDeque
# deque(['a', 'b', 'c'])
stackUsingDeque.popleft()  # popleft()
# 'a'
stackUsingDeque
# deque(['b', 'c'])
# Peek top element
stackUsingDeque[-1]
# 'c'

# QUEUE
# Init queue with empty list
QueueFromEmptyList = deque([])
# Append adds element to tail of the queue (Similar to what happens in queue DS)
QueueFromEmptyList.append(1)
QueueFromEmptyList.append(2)
QueueFromEmptyList.append(3)
# deque([1, 2, 3])
# Note: 1 is at the front of the queue, 3 is at the end of the queue
# Tail / Last element of queue, just like in arrays
QueueFromEmptyList[-1]
# 3
# Front / First element of queue
QueueFromEmptyList[0]
# 1
# Use popleft() to the front of the queue (Similar to what happens in queue DS)
QueueFromEmptyList.popleft()
# 1
QueueFromEmptyList
# deque([2, 3])
# Use pop() to the end of the queue (Never used ideally, because queue can only be popped from the front
QueueFromEmptyList.pop()
# 3
QueueFromEmptyList
# deque([2])

# append(): append to tail / end,
# appendleft(): append to head / front,
# pop(): pop from tail / end,
# popleft(): from from head
QueueFromEmptyList = deque([])
# Append adds element to tail of the queue (Similar to what happens in queue DS)
QueueFromEmptyList.append(1)
QueueFromEmptyList.append(2)
QueueFromEmptyList.append(3)


# Dictionary initialization
# Initialize map to int values
mapWithIntAsValue = defaultdict(int)
mapWithIntAsValue
# defaultdict(<class 'int'>, {})
# Initialize map to initial value field of all 1's
mapWithIntAsValueAll1s = defaultdict(lambda: 1)
mapWithIntAsValueAll1s
# defaultdict(<function <lambda> at 0x1038d9160>, {})
# Initialize map value fields to list / set
mapWithListAsValue = defaultdict(list)
mapWithListAsValue
# defaultdict(<class 'list'>, {})
mapWithSetAsValue = defaultdict(set)
mapWithSetAsValue
# defaultdict(<class 'set'>, {})
# Initialize defaultdict with a custom class
# https://www.geeksforgeeks.org/defaultdict-in-python/
# https://gist.github.com/poros/04a368f465e3c69d8d55
class Number(object):
    def __init__(self, N):
        self.N = N

    def __repr__(self):
        return str(self.N)

d = defaultdict(Number)
d['foo']
# TypeError: __init__() takes exactly 2 arguments (1 given)

d = defaultdict(lambda: Number(10))
d['foo']
# 10

# OR
from functools import partial
d = defaultdict(partial(Number, 10))
d['foo']
# 10


# HEAPS: Short summary
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
# Recursive Inorder + Heap: O(n*log(k))
def closestKValues(self, root: TreeNode, target: float, k: int) -> List[int]:
    # Initialize heap
    heap = []
    def inorder(r: TreeNode):
        if not r:
            return
        inorder(r.left)
        # Push to maxHeap (default is minHeap in Python)
        # Note: Inserts tuple (key, value). Key determines the order. Key is negative, because default is minHeap.
        heapq.heappush(heap, (-abs(r.val-target), r.val))
        if len(heap) > k:
            heapq.heappop(heap)
        inorder(r.right)

    inorder(root)
    return [x for _, x in heap]
# Peek() method: Take top element
# https://docs.python.org/3.1/library/heapq.html
# https://stackoverflow.com/questions/1750991/peeking-in-a-heap-in-python
import heapq
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 2)
heapq.heappush(heap, 5)
heap[0]
# 2
# Because this is min value in the min heap, which is the top element
heapq.heappop(heap)
# 2
heap[0]
# 3
# After popping top element, new top element is 3

# HEAPS
import heapq

# Initialize a heap (always a min-heap in Python) as a list
theHeap = []

# Push elements to min-heap
heapq.heappush(theHeap, 1)
heapq.heappush(theHeap, 10)
heapq.heappush(theHeap, 5)
# Print the contents of the heap, written from root to down
for element in theHeap:
    print(element)
'''
1
10
5
'''

# Remove the element at the root of the (min-)heap (minimum element)
heapq.heappop(theHeap)
'''
1
'''
# Print the contents of the heap, written from root to down
for element in theHeap:
    print(element)
'''
5
10
'''

# Push 9 on the heap, pop and return the smallest element of the heap
heapq.heappushpop(theHeap,9)
'''
5
'''
# Print the contents of the heap, written from root to down
for element in theHeap:
    print(element)
'''
9
10
'''

# Pop and return the smallest element from the heap, and then push the new item
heapq.heapreplace(theHeap,11)
'''
9
'''
# Print the contents of the heap, written from root to down
for element in theHeap:
    print(element)
'''
10
11
'''

# Create a heap from a list in-place, in-linear time
newHeap = [9,8,7,6,5,4,3,2,1]
heapq.heapify(newHeap)
# Print the contents of the heap, written from root to down
for element in newHeap:
    print(element)
'''
1
2
3
6
5
4
7
8
9
'''

# Get 4 smallest elements from the heap
smallestElements = heapq.nsmallest(4,newHeap)
print(smallestElements)
'''
[1, 2, 3, 4]

'''

# get 3 largest elements from the heap
largestElements = heapq.nlargest(3,newHeap)
print(largestElements)
'''
[9, 8, 7]
'''
print("Python Heap Example ends")
