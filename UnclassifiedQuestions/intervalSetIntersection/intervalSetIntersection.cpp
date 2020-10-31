// Given a set of intervals, determine if any two intervals overlap
// C++ solution: http://geeksquiz.com/check-if-any-two-intervals-overlap-among-a-given-set-of-intervals/
#include <iostream>
#include <algorithm>
using namespace std;

struct Interval
{
  int start;
  int end;
};

// Interval comparator: Compare intervals based on start time.
bool compareIntervalPair(Interval int1, Interval int2)
{
  return (int1.start < int2.start)? true : false ;
}

// Check if any two intervals overlap
bool isOverlap(Interval intervals[], int n)
{
  // 1. Sort intervals in increasing order of start time.
  sort(intervals, intervals+n-1, compareIntervalPair);

  // 2. Iterate over the sorted interval set (s_i < s_j for i < j).
  // For any interval pair i1=(s1,e1) and i2=(s2,e2), if s2 < e1,
  // then there is overlap.
  for(int i = 1; i < n; i++)
  {
    if(intervals[i-1].end > intervals[i].start)
      return true;
  }
  
  // If no pair in sorted list of intervals has an overlap, then there is no overlap.
  return false;
}

// Main driver
int main()
{
  // Example 1
  Interval arr1[] = { {1,3}, {7,9}, {4,6}, {10,13} };
  int n1 = sizeof(arr1) / sizeof(arr1[0]);
  cout << "Does the following interval list has an overlap?: {1,3}, {7,9}, {4,6}, {10,13} : ";
  isOverlap(arr1, n1) ? cout << "Yes\n" : cout << "No\n";

  // Example 2
  Interval arr2[] = { {6,8}, {1,3}, {2,4}, {4,7} };
  int n2 = sizeof(arr2) / sizeof(arr2[0]);
  cout << "Does the following interval list has an overlap?: {6,8},{1,3},{2,4},{4,7} : ";
  isOverlap(arr2, n2) ? cout << "Yes\n" : cout << "No\n";
}
