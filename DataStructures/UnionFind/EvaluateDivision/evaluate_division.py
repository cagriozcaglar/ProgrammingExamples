'''
Leetcode 399: Evaluate Division

You are given an array of variable pairs equations and an array of real numbers values, where equations[i] = [Ai, Bi]
and values[i] represent the equation Ai / Bi = values[i]. Each Ai or Bi is a string that represents a single variable.

You are also given some queries, where queries[j] = [Cj, Dj] represents the jth query where you must find the answer for Cj / Dj = ?.

Return the answers to all queries. If a single answer cannot be determined, return -1.0.

Note: The input is always valid. You may assume that evaluating the queries will not result in division by zero and that
there is no contradiction.

Note: The variables that do not occur in the list of equations are undefined, so the answer cannot be determined for them.
'''
from typing import List

class Solution:
    def calcEquation(
        self,
        equations: List[List[str]],
        values: List[float],
        queries: List[List[str]]
    ) -> List[float]:
        # Map from node to (groupid, weight) pair
        groupid_weight = {}

        def find(node_id):
            # Short version of the lines below:
            # group_id, node_weight = groupid_weight.setdefault(node_id, (node_id, 1))
            if node_id not in groupid_weight:
                groupid_weight[node_id] = (node_id, 1)
            group_id, node_weight = groupid_weight[node_id]

            # Inconsistency found, trigger chain update
            if group_id != node_id:
                new_group_id, group_weight = find(group_id)
                groupid_weight[node_id] = (new_group_id, node_weight * group_weight)
            return groupid_weight[node_id]

        def union(dividend, divisor, value):
            dividend_groupid, dividend_weight = find(dividend)
            divisor_groupid, divisor_weight = find(divisor)

            if dividend_groupid != divisor_groupid:
                # Merge two groups together, attaching dividend group to the one of advisor
                groupid_weight[dividend_groupid] = (divisor_groupid, divisor_weight * value / dividend_weight)

        # Step 1. Build union groups
        for (dividend, divisor), value in zip(equations, values):
            union(dividend, divisor, value)

        # Step 2. Run evaluation, with lazy updates in find() function
        results = []
        for (dividend, divisor) in queries:
            # Case 1. At least one variable didn't appear before
            if dividend not in groupid_weight or divisor not in groupid_weight:
                results.append(-1.0)
            # Case 2. Both variables appeared before
            else:
                dividend_groupid, dividend_weight = find(dividend)
                divisor_groupid, divisor_weight = find(divisor)
                # Case 2.1. Variables do not belong to the same group / chain
                if dividend_groupid != divisor_groupid:
                    results.append(-1.0)
                # Case 2.2. There is a chain / path between variables
                else:
                    results.append(dividend_weight / divisor_weight)

        return results