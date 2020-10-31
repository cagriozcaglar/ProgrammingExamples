import random


def findMedian(locations):
    def partition(arr, left, right):
        pivot = random.randint(left, right-1)
        arr[pivot], arr[right] = arr[right], arr[pivot]
        p = left
        for i in range(left, right):
            if arr[i] < arr[right]:
                arr[i], arr[p] = arr[p], arr[i]
                p = p+1
        arr[p], arr[right] = arr[right], arr[p]
        return p

    l, r = 0, len(locations)-1
    mid = int((l+r) / 2)
    while True:
        p = partition(locations, l, r)
        print(p)
        if p == mid:
            return locations[mid] if len(locations) % 2 != 0 else (locations[mid] + locations[mid+1]) / 2.0
        if p < mid:
            l = p+1
        else:
            r = p-1


def main():
    #locations = [4,5,6,3,2]
    locations = [4,5,6,3,2,1]
    num = findMedian(locations)
    print("num: " + str(num))

if __name__== "__main__":
    main()