import java.util.*;

public class pairExample
{
    public static void main(String[] args)
    {
	Map.Entry<String, String> thePair = new AbstractMap.SimpleEntry<String,String>("A","B");
	System.out.println(thePair.getKey() + " , " + thePair.getValue());
	Map.Entry<Integer,Integer> thePair2 = new AbstractMap.SimpleEntry<Integer,Integer>(1,2);
	System.out.println(thePair2.getKey().toString() + " , " + thePair2.getValue().toString());
    }
}
