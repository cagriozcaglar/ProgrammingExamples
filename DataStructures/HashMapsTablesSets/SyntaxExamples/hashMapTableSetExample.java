import java.util.*;

public class hashMapTableSetExample
{
    public static void main(String[] args)
    {
		//////////////////////////// 1. HashMap ////////////////////////////
		System.out.println("HashMaps:");
		HashMap<Integer, String> integerStringMap = new HashMap<Integer, String>();
		integerStringMap.put(1, "One");
		integerStringMap.put(2, "Two");
		integerStringMap.put(3, "Three");

		System.out.println(integerStringMap.get(2));
		System.out.println(integerStringMap.containsKey(3));
		System.out.println(integerStringMap.isEmpty());
		System.out.println(integerStringMap.size());

		integerStringMap.remove(3);
		System.out.println(integerStringMap.size());
		integerStringMap.put(3, "Three");

		System.out.println(integerStringMap.keySet());
		System.out.println(integerStringMap.entrySet());

		// Print HashMap: Method 1
		Iterator it = integerStringMap.entrySet().iterator();
		while(it.hasNext())
		{
			Map.Entry entry = (Map.Entry)it.next();
			System.out.print("(" + entry.getKey() + ", " + entry.getValue() + ") ");
		}
		System.out.println();

		// Print HashMap: Method 2
		for(Map.Entry<Integer,String> entry : integerStringMap.entrySet())
			System.out.print("(" + entry.getKey() + ", " + entry.getValue() + ") ");
		System.out.println();

		// Print HashMap: Method 3
		System.out.println(integerStringMap.entrySet() + "\n");

		//////////////////////////// 2. HashTable ////////////////////////////
		System.out.println("Hashtables:");
		Hashtable<Integer, String> integerStringTable = new Hashtable<Integer, String>();
		integerStringTable.put(1, "One");
		integerStringTable.put(2, "Two");
		integerStringTable.put(3, "Three");

		System.out.println(integerStringTable.get(2));
		System.out.println(integerStringTable.containsKey(3));
		System.out.println(integerStringTable.isEmpty());
		System.out.println(integerStringTable.size());

		integerStringTable.remove(3);
		System.out.println(integerStringTable.size());
		integerStringTable.put(3, "Three");

		System.out.println(integerStringTable.keySet());
		System.out.println(integerStringTable.entrySet());

		// Print Hashtable: Method 1
		Iterator it2 = integerStringTable.entrySet().iterator();
		while(it2.hasNext())
		{
			Map.Entry entry = (Map.Entry)it2.next();
			System.out.print("(" + entry.getKey() + ", " + entry.getValue() + ") ");
		}
		System.out.println();

		// Print Hashtable: Method 2
		for(Map.Entry<Integer,String> entry : integerStringTable.entrySet())
			System.out.print("(" + entry.getKey() + ", " + entry.getValue() + ") ");
		System.out.println();

		// Print Hashtable: Method 3
		System.out.println(integerStringTable.entrySet());

		// Print Hashtable: Method 4
		Enumeration keys = integerStringTable.keys();
			int key;
		while(keys.hasMoreElements())
		{
			key = (Integer)keys.nextElement();
			System.out.print("(" + key + ", " + integerStringMap.get(key) + ") ");
		}
		System.out.println("\n");

		//////////////////////////// 3. HashSet ////////////////////////////
		System.out.println("Hashsets:");
		HashSet<String> stringSet = new HashSet<String>();
		stringSet.add("One");
		stringSet.add("Two");
		stringSet.add("Three");
		System.out.println(stringSet);

		stringSet.remove("Two");
		System.out.println(stringSet);
		stringSet.add("Two");
		System.out.println(stringSet);

		System.out.println(stringSet.size());
		System.out.println(stringSet.isEmpty());
		System.out.println(stringSet.contains("One"));

		// Print: Method 1
		System.out.println(stringSet);

		// Print: Method 2
		Iterator it3 = stringSet.iterator();
		while(it3.hasNext())
		{
			System.out.print( (String)it3.next() + " ");
		}
		System.out.println();

		// Print: Method 3
		for(String stringElement : stringSet)
			System.out.print( stringElement + " ");
		System.out.println("\n");
    }
}
