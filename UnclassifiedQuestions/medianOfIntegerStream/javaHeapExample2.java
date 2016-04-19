import java.util.PriorityQueue;

public class javaHeapExample2
{
    public static void main(String[] args)
    {
	PriorityQueue<Request> queue = new PriorityQueue<>();
	queue.offer(new Request("ABC", 2));
	queue.offer(new Request("ABC", 5));
	queue.offer(new Request("ABC", 1));
	while(!queue.isEmpty())
	{
	    System.out.println(queue.poll());
	}
    }
    /* Output:
       Request [requestName= ABC, priorityStatus=1]
       Request [requestName= ABC, priorityStatus=2]
       Request [requestName= ABC, priorityStatus=5]
    */
}

class Request implements Comparable<Request>
{
    private String requestName = "";
    private int priorityStatus = 0;

    public Request(String requestName, int priorityStatus)
    {
	this.requestName = requestName;
	this.priorityStatus = priorityStatus;
    }

    @Override
    public int compareTo(Request otherRequest)
    {
	return Integer.compare(priorityStatus, otherRequest.priorityStatus);
    }

    @Override
    public String toString()
    {
	return "Request [requestName= " + requestName + ", priorityStatus=" + priorityStatus + "]";
    }
}
