/*
Given an array of integers, return three elements that sum to 0.
If there is no such triple, return an error message.
*/

/*
e.g. A = [3, 2, -4,  6 , 7, -8] => 2 + 6 + (-8) = 0
*/

/* Design:
// a + b = k     // 2-sum
// Sort the array
// Keep a pointer at the beginning and end

// a + b + c = 0  // 3-sum 
// a + b = -c
// a + b = k      // Similar to 2-sum

# Solution:
1. Get all pairs in the array and map their sum to all possible pairs that sum to the value: O(n^2)
2. Iterate over the array, and for each element A[i], check the hashmap to see if there is a pair that sum up to A[-i]: O(n) best case, O(n^2) worst case.
*/
import java.util.*;

public class ThreeSum
{
    public static int[] getThreesum(int[] A)
    {
	HashMap<Integer, Vector< Pair<int,int> > > sumToPairs = new HashMap<Integer, Pair<int,int>>();
	int sum;
	int[] threeSum;
  
	// Create pairs (a,b) => O(n^2)
	// Create hashmap from sums to all pairs that sum to that value // O(n^2)
	for(int i=0; i < A.length(); i++)
	{
	    for(int j=i+1; j < A.length(); j++)
	    {
		sum = A[i] + A[j]
		if(!sumToPairs.containsKey(sum))
		{
		    Vector< Pair<int,int> > newPairVector = new Vector< Pair<int,int> >();
		    newPairVector.put(Pair(i,j))
		    sumToPairs.put(sum, newPairVector)
		}
		else
		{
		    sumToPairs[sum].put(Pair(i,j))
		}
	    }
	}  

	// pair.first, pair.second, -A[i]  
	// Iterate over array: O(n)
	for(int i=0; i < A.length(); i++ )
	{
	    if(sumToPairs.containsKey(-A[i]))
	    {
		//Pair sumPairIndices = sumToPairs[-A[i]]
		for(Pair<int,int> sumPairIndices : sumToPairs[-A[i]])
		{
		    if(sumPairIndices.first != i && sumPairIndices.second != i)
		    {
			threeSum = {A[sumPair.first], A[sumPair.second], -A[i]}
			return threeSum;
		    }
		}
	    }
	}
	return threeSum;
    }

    public static void main(String[] args)
    {
	
    }
}
