import java.util.*;

/*
C++ solution: http://geeksquiz.com/check-if-any-two-intervals-overlap-among-a-given-set-of-intervals/
http://www.programcreek.com/2012/12/leetcode-insert-interval/
https://www.careercup.com/question?id=13014685
http://codereview.stackexchange.com/questions/113655/counting-the-overlapping-intervals-in-the-union-of-two-sets
*/

public class IntervalSetIntersection
{
    public static void main(String[] args)
    {
	Vector<Interval> intervals = new Vector<Interval>();
	intervals.add(new Interval(1.2,5.3));
	intervals.add(new Interval(4.3,8.1));
	intervals.add(new Interval(9.2,6.4));
	
	for(Interval interval : intervals)
	{
	    interval.print();
	}

    }

    // Interval class
    class Interval
    {
	private double start;
	private double end;

	public Interval(double start, double end)
	{
	    this.start = start;
	    this.end = end;
	}

	public double getStart()
	{
	    return this.start;
	}

	public double getEnd()
	{
	    return this.end;
	}

	public void print()
	{
	    System.out.println("(" + start + ", " + end + ")");
	}

	/*    
	public static void main(String[] args)
	{
	    Interval int1 = new Interval(1.2, 3.4);
	    Interval int2 = new Interval(5.6, 7.8);
	    int1.print();
	    int2.print();
	}
	*/
    }

}
