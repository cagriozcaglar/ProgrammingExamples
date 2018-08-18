import unittest

def SumOfSquares(c):
    """
    Inner function: Check if number n is a square number 
    :param self: 
    :param c: 
    :return: 
    """
    def isSquare(n):
        # Check if (sqrt(n))^2 is equal to n, which means n is a square
        return ( (n ** 0.5) ** 2 ) == n
    return any( isSquare(c - (num ** 2)) for num in xrange( int(c ** 0.5) +1 ) )

if __name__ == "__main__":
    # Test 1: True (10 = 1^2 + 3^2)
    x = 10
    print(SumOfSquares(x))

    # Test 2: True (8 = 2^2 + 2^2)
    x = 8
    print(SumOfSquares(x))

    # Test 3: False (11)
    x = 11
    print(SumOfSquares(x))