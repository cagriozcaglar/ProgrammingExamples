'''
Leetcode 38: Count and Say

The count-and-say sequence is a sequence of digit strings defined by the recursive formula:

countAndSay(1) = "1"
countAndSay(n) is the way you would "say" the digit string from countAndSay(n-1), which is then converted into a different digit string.

To determine how you "say" a digit string, split it into the minimal number of groups so that each group is a contiguous section all of the same character. Then for each group, say the number of characters, then say the character. To convert the saying into a digit string, replace the counts with a number and concatenate every saying.

For example, the saying and conversion for digit string "3322251":

Given a positive integer n, return the nth term of the count-and-say sequence.

Example 1:
Input: n = 1
Output: "1"
Explanation: This is the base case.

Example 2:
Input: n = 4
Output: "1211"
Explanation:
countAndSay(1) = "1"
countAndSay(2) = say "1" = one 1 = "11"
countAndSay(3) = say "11" = two 1's = "21"
countAndSay(4) = say "21" = one 2 + one 1 = "12" + "11" = "1211"
'''

class Solution:
    def runLengthEncoding(self, rel_val: str) -> str:
        if not rel_val:
            return ""
        result = []
        current_value = rel_val[0]
        count = 1
        for i in range(1, len(rel_val)):
            # If previous value continues, increment count, do not write
            if rel_val[i] == rel_val[i-1]:
                count += 1
            else:
                result.append(f"{str(count)}{current_value}")
                current_value = rel_val[i]
                count = 1
        # IMPORTANT: You forgot this line.
        result.append(f"{str(count)}{current_value}")
        return "".join(result)

    def countAndSay(self, n: int) -> str:
        rle_val = "1"  # rle("1") = "1"
        count = 2
        while count <= n:
            rle_val = self.runLengthEncoding(rle_val)
            count += 1
        return rle_val

solution = Solution()
solution.countAndSay(1)
solution.countAndSay(2)
solution.countAndSay(3)
solution.countAndSay(4)