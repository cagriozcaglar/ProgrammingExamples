"""
Use stack for "{" and "}" just like in calculator.
Maintain two lists:
1. the previous list before ",",
2. the current list that is still growing.

Cases:
1. When seeing an "alphabet", grow the second list by corss multiplying. So that [a]*b will become "ab", [a,c]*b will become ["ab", "cb"]
2. When seeing ",", that means the second list can't grow anymore. combine it with the first list and clear the second list.
3. When seeing "{", push the two lists into stack,
4. When seeing "}", pop the previous two lists, cross multiplying the result inside "{ }" into the second list.

If not considering the final sorting before return, the time complexity should be O(n)
"""
from itertools import product

class Solution:
    def braceExpansionII(self, expression: str) -> List[str]:
        # 1. Stack to do calculations
        # 2. Result list to hold previous list before ","
        # 3. Current list to hold current list
        stack, result, current = [], [], []
        for v in expression:
            # 1. v is alpha-numeric
            if v.isalpha():
                # Concatenation with cross product
                # If current is non-empty, use cross-product. Else, push v to current.
                current = [c+v for c in current] if current else [v]
            # 2. v is "{"
            # Start of expression. Push the two lists result and current to stack, and reset them
            elif v == '{':
                stack.append(result)
                stack.append(current)
                result, current = [], []
            # 3. v is "}"
            # End of expression: Pop previous two lists from stack, cross-multiply the result inside "{}" into the second list
            elif v == '}':
                preCurrent = stack.pop()
                preResult = stack.pop()
                # cartesianProduct = list[product(result+current, preCurrent)]
                # current = [a+b for a,b in cartesianProduct] if cartesianProduct else ['']
                current = [p+c for c in result+current for p in preCurrent or ['']]
                result = preResult
            # 4. v is ","
            # Second list cannot grow anymore. Combine with first list and clear the second list
            elif v==',':
                result += current
                current = []
        return sorted(set(result+current))