"""
Given two sorted lists of non-null integers sorted in ascending order, 
return the intersection and union of these lists as a new list in ascending order.
@param a: A list of non-null integers, in ascending order
@param b: A list of non-null integers, in ascending order
@return two lists: 
  1) Union: The integers that are contained in either a or b, in ascending order
  2) Intersection: The integers that are contained in a and b, in ascending order

Note: If the input lists have duplicates, union list will have as many as 
the number of times the duplicate value appears in both lists. For intersection list,
display the element only once.
"""

def unionAndIntersection(listA, listB):
    unionList = []
    intersectionList = []
    i,j = 0,0
    while(i < len(listA) and j < len(listB)):
        # Value in listA is smaller than value in listB
        if(listA[i] < listB[j]):
            unionList.append(listA[i])
            i += 1
        # Value in listA is equal to value in listB
        elif listA[i] == listB[j]:
            unionList.append(listA[i])
            unionList.append(listB[j])
            # If the list is empty, or if the list is non-empty and last element of the list is different from listA[i]
            # Then, add the element to intersection list
            if( (not intersectionList) or (intersectionList and listA[i] != intersectionList[-1])):
                intersectionList.append(listA[i])
            i += 1
            j += 1
        # Value in listA is greater than value in listB
        else:
            unionList.append(listB[j])
            j += 1

    # If one or both of the lists are not complete, add their elements to result list
    while i < len(listA):
        unionList.append(listA[i])
        i += 1
    while j < len(listB):
        unionList.append(listB[j])
        j += 1

    return [unionList,intersectionList]

if __name__ == "__main__":
    # Test 1
    a = [1,2,3]
    b = [1,2,3,4]
    [union, intersection] = unionAndIntersection(a,b)
    print(union)
    print(intersection)
    # Test 2
    a = [1,1,2,2,3]
    b = [1,1,2,2,3,3,4]
    [union, intersection] = unionAndIntersection(a,b)
    print(union)
    print(intersection)
    # Test 3
    a = [1,2,3]
    b = [4,5,6,7]
    [union, intersection] = unionAndIntersection(a,b)
    print(union)
    print(intersection)
    # Test 4
    a = [1,2,3,4,4]
    b = [4,5,6]
    [union, intersection] = unionAndIntersection(a,b)
    print(union)
    print(intersection)
    # Test 4
    a = [1,2,3,4,4]
    b = [3,4,4,5,6]
    [union, intersection] = unionAndIntersection(a,b)
    print(union)
    print(intersection)
