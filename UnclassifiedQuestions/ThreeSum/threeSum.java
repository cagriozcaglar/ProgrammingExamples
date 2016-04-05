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

// Use Map.Entry<Integer, Integer> in Java to represent an Integer pair.
// Example: Map.Entry<String, String> thePair = new AbstractMap.SimpleEntry<>("A","B");

public class threeSum
{
    public static int[] getThreeSum(int[] A)
    {
	HashMap<Integer, Vector< Map.Entry<Integer,Integer> > > sumToPairs = new HashMap<Integer, Vector< Map.Entry<Integer,Integer> > >();
	int sum;
	int[] threeSum;
  
	// Create pairs (a,b) => O(n^2)
	// Create hashmap from sums to all pairs that sum to that value // O(n^2)
	for(int i=0; i < A.length; i++)
	{
	    for(int j=i+1; j < A.length; j++)
	    {
		sum = A[i] + A[j];
		if(!sumToPairs.containsKey(sum))
		{
		    Vector< Map.Entry<Integer,Integer> > newPairVector = new Vector< Map.Entry<Integer,Integer> >();
		    newPairVector.addElement( new AbstractMap.SimpleEntry<Integer,Integer>(i,j) );
		    sumToPairs.put(sum, newPairVector);
		}
		else
		{
		    sumToPairs.get(sum).addElement( new AbstractMap.SimpleEntry<Integer,Integer>(i,j) );
		}
	    }
	}  

	// pair.first, pair.second, -A[i]
	// Iterate over array: O(n)
	for(int i=0; i < A.length; i++ )
	{
	    if(sumToPairs.containsKey(-A[i]))
	    {
		//Pair sumPairIndices = sumToPairs[-A[i]]
		for(Map.Entry<Integer,Integer> sumPairIndices : sumToPairs.get(-A[i]))
		{
		    Integer firstIndex = sumPairIndices.getKey();
		    Integer secondIndex = sumPairIndices.getValue();
		    if(firstIndex != i && secondIndex != i)
		    {
			threeSum = new int[]{A[firstIndex], A[secondIndex], -A[i]};
			return threeSum;
		    }
		}
	    }
	}
	threeSum = new int[]{};
	return threeSum;
    }

    public static void main(String[] args)
    {
	// First test
	int[] integerArray = new int[] {3, 2, -4, 6, 7, -8};
	System.out.println(Arrays.toString(integerArray));
	int[] threeSum = getThreeSum(integerArray);
	System.out.println("A three-element subset of array that sum up to zero: " + Arrays.toString(threeSum));
	/* Output:
	   [3, 2, -4, 6, 7, -8]
	   A three-element subset of array that sum up to zero: [6, -8, -2]
	*/

	// Second test
	int[] integerArray2 = new int[] {3, 2, -4, 6, 7};
	System.out.println(Arrays.toString(integerArray2));
	int[] threeSum2 = getThreeSum(integerArray2);
	System.out.println("A three-element subset of array that sum up to zero: " + Arrays.toString(threeSum2));
	/*
	  [3, 2, -4, 6, 7]
	  A three-element subset of array that sum up to zero: []
	*/
    }
}
