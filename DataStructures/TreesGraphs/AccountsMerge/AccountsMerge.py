"""
721. Accounts Merge
(DFS, or UF (not implemented in this one))
Given a list of accounts where each element accounts[i] is a list of strings, where the first element accounts[i][0] is
a name, and the rest of the elements are emails representing emails of the account.

Now, we would like to merge these accounts. Two accounts definitely belong to the same person if there is some common
email to both accounts. Note that even if two accounts have the same name, they may belong to different people as people
could have the same name. A person can have any number of accounts initially, but all of their accounts definitely have
the same name.

After merging the accounts, return the accounts in the following format: the first element of each account is the name,
and the rest of the elements are emails in sorted order. The accounts themselves can be returned in any order.
"""
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        visited = [False] * len(accounts)
        emailsToAccountIdsMap = defaultdict(list)
        result = []
        for i, account in enumerate(accounts):
            for email in account[1:]:
                emailsToAccountIdsMap[email].append(i)

        # DFS for traversing account
        def dfs(i, emails):
            if visited[i]:
                return
            visited[i] = True
            # Add each email in account to emails set
            for email in accounts[i][1:]:
                emails.add(email)
                # For each neighbour account Id (to email), call dfs again
                for neighAccountId in emailsToAccountIdsMap[email]:
                    dfs(neighAccountId, emails)

        # Perform DFS for accounts and add to results
        for i, account in enumerate(accounts):
            if visited[i]:
                continue
            name, emails = account[0], set()
            dfs(i, emails)
            result.append([name] + sorted(emails))

        return result