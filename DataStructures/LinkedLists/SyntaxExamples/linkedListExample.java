import java.util.*;

public class linkedListExample
{
    public static void main(String[] args)
    {
	LinkedList<String> linkedList = new LinkedList<String>();
	linkedList.add("One");
	linkedList.add("Two");
	linkedList.add("Three");
	System.out.println(linkedList.toString());

	linkedList.addFirst("Zero");
	linkedList.addLast("Four");

	System.out.println(linkedList.get(0));

	linkedList.removeFirst();
	linkedList.removeLast();
	System.out.println(linkedList.toString());
	
	linkedList.remove(1);
	System.out.println(linkedList.toString());
	System.out.println(linkedList.size() + "\n");
    }
}
