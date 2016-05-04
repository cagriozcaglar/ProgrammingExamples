import java.util.*;

public class javaHeapExample0{
    public static void main(String[] args)
    {
	System.out.println("\nJava Heap Example 0 starts");
	Queue<Integer> minHeap = new PriorityQueue<Integer>();
	Queue<Integer> maxHeap = new PriorityQueue<Integer>(10, new maxHeapComparator());

	System.out.println("Min Heap:");
	// Add to minHeap
	minHeap.add(3);
	minHeap.add(2);
	minHeap.add(1);
	System.out.println(minHeap.toString());
	// Output: [1, 3, 2]
	// Peek from minHeap
	System.out.println(minHeap.peek());
	System.out.println(minHeap.toString());
	// Poll from minHeap
	minHeap.poll();
	System.out.println(minHeap.toString());
	// Check if contains
	System.out.println(minHeap.contains(3));
	// Get size
	System.out.println(minHeap.size());
	// Add root and print using iterator
	minHeap.add(1);
	Iterator it = minHeap.iterator();
	while(it.hasNext())
	{
	    System.out.print(it.next() + " ");
	}
	// Output: 1 3 2
	System.out.println();

	System.out.println("Max Heap:");
	// Add to maxHeap
	maxHeap.add(3);
	maxHeap.add(2);
	maxHeap.add(1);
	System.out.println(maxHeap.toString());
        // Output: [3, 2, 1]
	// Peek from minHeap
	System.out.println(maxHeap.peek());
	System.out.println(maxHeap.toString());
	// Poll from minHeap
	maxHeap.poll();
	System.out.println(maxHeap.toString());
	// Check if contains
	System.out.println(maxHeap.contains(2));
	// Get size
	System.out.println(maxHeap.size());
	// Add root and print using iterator
	maxHeap.add(3);
	Iterator it2 = maxHeap.iterator();
	while(it2.hasNext())
	{
	    System.out.print(it2.next() + " ");
	}
	// Output: 3 2 1
	System.out.println();

	System.out.println("Java Heap Example 0 ends\n");
    }

    private static class maxHeapComparator implements  Comparator<Integer>
    {
	@Override
        public int compare(Integer o1, Integer o2)
	{
	    return o2-o1;
	}
    }
}
