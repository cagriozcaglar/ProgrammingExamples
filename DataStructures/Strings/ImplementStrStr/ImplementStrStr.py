class ImplementStrStr:
    # Solution 2: Robin Karp Algorithm (Uses rolling hash)
    # Time complexity: O(n+m) average case
    # Adopted from here: https://lenchen.medium.com/leetcode-28-implement-strstr-64de75d9ffb1
    def strStr(self, haystack: str, needle: str) -> int:
        # Base: randomly choose big prime
        base = 5  # 13  # 401

        # String lengths
        m = len(haystack)
        n = len(needle)

        # Hash function
        def hashing(S):
            s = len(S)
            return sum(
                [
                    ord(S[i]) * (base ** (s-i-1)) for i in range(s)
                ]
            )

        # Rolling hash function: Calculate new hash from old
        # new_value = (H - (a_1)x^(n-1)) * x + a_(n+1)
        def rolling_hashing(old_hash, old_value, new_value):
            #return base * (old_hash - (base ** (n-1)) * ord(old_value)) + ord(new_value)
            return (old_hash - ord(old_value) * (base ** (n-1))) * base + ord(new_value)

        # Algorithm starts now
        # Get hash value for pattern and text at i=0
        n_hash = hashing(needle)
        m_hash = hashing(haystack[0:n]) # get hash for first substring

        # Check if pattern matches first substring
        if m_hash == n_hash:
            return 0

        for i in range(1, m-n+1):
            m_hash = rolling_hashing(m_hash, haystack[i-1], haystack[i+n-1])
            if m_hash == n_hash:
                return i

        return -1