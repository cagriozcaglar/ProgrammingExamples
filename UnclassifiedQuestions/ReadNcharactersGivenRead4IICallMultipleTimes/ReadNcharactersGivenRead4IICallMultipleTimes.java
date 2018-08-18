/**
 The API: int read4(char *buf) reads 4 characters at a time from a file.
 The return value is the actual number of characters read. For example, it returns 3 if there is only 3 characters left in the file.
 By using the read4 API, implement the function int read(char *buf, int n) that reads n characters from the file.

 Note: The read function may be called **multiple** times.
 */

// Solution: http://buttercola.blogspot.com/2014/11/leetcode-read-n-characters-given-read4_23.html

public ReadNcharactersGivenRead4IICallMultipleTimes {
    /*
    This makes the problem a lot more complicated, because it can be called multiple times
    and involves storing states.
    Therefore, we design the following class member variables to store the states:
     i. buffer – An array of size 4 use to store data returned by read4 temporarily. If the characters were read into the
        buffer and were not used partially, they will be used in the next call.
     ii. offset – Use to keep track of the offset index where the data begins in the next read call. The buffer could be
         read partially (due to constraints of reading up to n bytes) and therefore leaving some data behind.
     iii. buffSize – The real buffer size that stores the actual data. If buffSize > 0, that means there is partial data left
          in buffer from the last read call and we should consume it before calling read4 again. On the other hand, if
          bufsize == 0, it means there is no data left in buffer.
     This problem is a very good coding exercise. Coding it correctly is extremely tricky due to the amount of edge cases to consider.
    */
    private char[] buffer = new char[4];
    int offset = 0;
    int buffSize = 0;

    /**
     *
     * @param buf Destination buffer
     * @param n   Maximum number of characters to read
     * @return    The number of characters read
     */
    public int read(char[] buf, int n) {
        int readBytes = 0;
        boolean eof = false;

        while(!eof && readBytes < n) {
            int currentSize = (buffSize > 0) ? buffSize : read4(buffer);
            if(buffSize == 0 && currentSize < 4) {
                eof = true;
            }
            int bytes = Math.min(n-readBytes, currentSize);
            // This is the key array copy step: buffer -> buf
            System.arraycopy(buffer /* src */, offset /* srcPos */, buf /* dest */, readBytes /* destPos */, bytes /* length */);
            offset = (offset + bytes) % 4;
            buffSize = currentSize - bytes;
            readBytes += bytes;
        }
        return readBytes;
    }
}

/*
This problem is not very hard, but requires thinking of every corner cases. To sum up, the key of the problem is to put the char buf4[4] into global, and maintains two more global variables:
 -- offset : the starting position in the buf4 that a read() should start from.
 -- bytesLeftInBuf4 : how many elements left in the buf4.
One corner case to consider is when is the eof should be true? In the previous question, it is true only if bytesFromRead4 < 4. However, in this question, since we might have some bytes left the buf4, even if it is not end of the file, we may mistakely consider the eof as true. So the condition to set eof is true is bytesFromRead4 < 4 && bytesLeftInBuf4 == 0
Another corner case we need to consider is: if the bytesFromRead4 + bytesRead > n, the actual bytes to copy is n - bytesRead.
For example, the file is "abcde", and read(2), in this case, the bytesFromRead4 = 4, but we should only copy 2 bytes from the buf4. So be very careful about this case.
At the end, we need to update the global offset and bytesLeftInBuf4, as well as the local bytesRead.
 */