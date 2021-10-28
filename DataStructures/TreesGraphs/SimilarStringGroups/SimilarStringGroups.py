"""
839. Similar String Groups

Two strings X and Y are similar if we can swap two letters (in different positions) of X, so that it equals Y.
Also two strings X and Y are similar if they are equal.

For example, "tars" and "rats" are similar (swapping at positions 0 and 2), and "rats" and "arts" are similar, but
"star" is not similar to "tars", "rats", or "arts".

Together, these form two connected groups by similarity: {"tars", "rats", "arts"} and {"star"}.  Notice that "tars"
and "arts" are in the same group even though they are not similar. Formally, each group is such that a word is in the
group if and only if it is similar to at least one other word in the group.

We are given a list strs of strings where every string in strs is an anagram of every other string in strs.
How many groups are there?
"""
class Solution:
    def numSimilarGroups(self, strs: List[str]) -> int:
        # Generate adjacency graph among strings
        adj = defaultdict(list)
        for i, string in enumerate(strs):
            # Careful: End index is len(strs), because it is exclusive range boundary
            for j in range(i+1, len(strs)):
                if self.isSimilar(strs[i], strs[j]):
                    adj[strs[i]].append(strs[j])
                    adj[strs[j]].append(strs[i])
        visited = set()

        def dfs(word) -> None:
            nonlocal adj
            nonlocal visited
            if word in visited:
                return
            visited.add(word)
            for simWord in adj[word]:
                dfs(simWord)

        count = 0
        for word in strs:
            if word not in visited:
                dfs(word)
                count += 1
        return count

    def isSimilar(self, str1: str, str2: str) -> bool:
        diff = 0
        for ch1, ch2 in zip(str1, str2):
            if ch1 != ch2:
                diff += 1
        return diff <= 2