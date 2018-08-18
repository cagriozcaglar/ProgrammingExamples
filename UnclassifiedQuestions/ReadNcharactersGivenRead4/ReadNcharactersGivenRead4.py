'''
The API: int read4(char *buf) reads 4 characters at a time from a file.
The return value is the actual number of characters read. For example, it returns 3 if there is only 3 characters left in the file.
By using the read4 API, implement the function int read(char *buf, int n) that reads n characters from the file.

Note: The read function will only be called **once** for each test case.
'''

# Clean solution: https://discuss.leetcode.com/topic/108186/clean-and-straightforward-solution
# Another solution: https://tenderleo.gitbooks.io/leetcode-solutions-/GoogleEasy/157.html
# Java Solution: https://github.com/yaq3516/Leetcode-locked/blob/master/157%20Read%20N%20Characters%20Given%20Read4.java
def read(buf, n):
    r = 4
    current = 0
    while current < sum and r==4:
        temp = []
        r = read4(temp)
        # Get minimum of r==4 and remaining characters (n-current)
        numCharsRead = min(n-current, r)
        # Copy chars from temp to buffer
        for i in range(len(temp)):
            buf[i] = temp[i]
        # Added number of characters read
        current += numCharsRead
    return current

if __name__ == "__main__":
    temp = [1,2,3,4,5,6,7,8,9,10]
    print("read(temp, 5): " + read(temp, 5))