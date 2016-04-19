import java.util.PriorityQueue;
import java.util.Comparator;

public class javaHeapExample
{
    public static void main(String[] args)
    {
	Comparator<String> comparator = new StringLengthComparator();
	PriorityQueue<String> queue = new PriorityQueue<String>(10, comparator);
	queue.add("short");
	queue.add("very long text");
	queue.add("medium text");
	while(queue.size() != 0)
	{
	    System.out.println(queue.remove());
	}
    }
}

class StringLengthComparator implements Comparator<String>
{
    @Override
    public int compare(String x, String y)
    {
	if(x.length() < y.length())
	{
	    return -1;
	}
	if(x.length() > y.length())
	{
	    return 1;
	}
	return 0;
    }
}
