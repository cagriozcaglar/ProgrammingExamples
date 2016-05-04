import java.util.*;

public class stackQueueExample
{
    public static void main(String[] args)
    {
	// Stack
	System.out.println("Stacks:");
	Stack<Integer> intStack = new Stack<Integer>();
	intStack.push(1);
	intStack.push(2);
	intStack.push(3);
	System.out.println(intStack);

	intStack.pop();
	System.out.println(intStack);
	intStack.push(3);
	System.out.println(intStack);
	// Safe pop() with exception check
	try{
	    intStack.pop();
	}
	catch (EmptyStackException e){
	    System.out.println("empty stack");
	}

	System.out.println(intStack.peek());
	System.out.println(intStack);

	System.out.println(intStack.isEmpty());
	System.out.println(intStack.empty());
	System.out.println(intStack.size());

	// Queue
	System.out.println("Queues:");
	// Queue is an interface in Java. You can use LinkedList or PriorityQueue to instantiate a queue.
	Queue<Integer> queue1 = new LinkedList<Integer>();
	Queue<Integer> queue2 = new PriorityQueue<Integer>();

	// Enqueue: Using add() method
	queue1.add(1);
	queue1.add(2);
	queue1.add(3);
	System.out.println(queue1);
	System.out.println(queue2);
	
	// Dequeue: using remove() method
	System.out.println(queue1.remove());  // Remove the last element of the queue
	System.out.println(queue1.size());    // Get the size of the queue
	System.out.println(queue1);
	queue1.add(1);                      // 1 is inserted at the end of the queue
	System.out.println(queue1.size());  // Get the size of the queue
	System.out.println(queue1);

	// Peek: using element() method
	System.out.println(queue1);
	System.out.println(queue1.element()); // Peek: get the value of the first element in the queue.
	System.out.println(queue1);

	System.out.println(queue1.isEmpty() + "\n");
    }
}
