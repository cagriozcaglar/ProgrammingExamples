"""
158. Read N Characters Given Read4 II - Call multiple times
Given a file and assume that you can only read the file using a given method read4, implement a method read to read n
characters. Your method read may be called multiple times.
"""
# The read4 API is already defined for you.
# def read4(buf4: List[str]) -> int:

class Solution:
    def __init__(self):
        self.queue = collections.deque()

    def read(self, buf: List[str], n: int) -> int:
        buf4 = [''] * 4
        total_read_so_far = 0
        while total_read_so_far < n:
            # Try to read 4 characters from the file
            numChars = read4(buf4)
            # Increment the count so we can stop when we have read enough
            total_read_so_far += numChars
            # Add only the characters read into the deque, since buf4 has 4 characters in it, some possibly from before
            self.queue.extend(buf4[:numChars])
            # We want to read n characters, but we have run out of characters in the file!
            # Escape the loop to prevent getting stuck in infinite loop
            if numChars == 0:
                break

        # queue has the required characters, read n of them, or whatever q has if it
        # is less than n and put it in the buffer provided starting from 0
        i = 0
        while i < n and self.queue:
            buf[i] = self.queue.popleft()
            print(buf[i])
            i += 1

        return i

    def read2(self, buf: List[str], n: int) -> int:
        """
        buf: Destination buffer
        n: number of characters to read
        """
        total = 0
        tempBuffer = [''] * 4
        while total < n:
            # Queue is empty
            if not self.queue:
                size = read4(tempBuffer)
                if not size:
                    break
                self.queue.extend(tempBuffer[:size])
            else:
                buf[total] = self.queue.popleft()
                total += 1
        return total