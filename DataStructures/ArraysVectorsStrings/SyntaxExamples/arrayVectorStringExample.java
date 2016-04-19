import java.util.*;

public class arrayVectorStringExample
{
    public static void main(String[] args)
    {
	/////////////////////////// 1. Arrays //////////////////////////
	System.out.println("Arrays:");
	int[] integerArray = {1,2,3,4,5};
	System.out.println( Arrays.toString(integerArray) );
	
	// Print Array: Method 1
	for(int i=0; i < integerArray.length; i++) // length, not length()
	{
	    System.out.print(integerArray[i] + " ");
	}
	System.out.println();
	// Print Array: Method 2
	for(int anInt : integerArray)
	{
	    System.out.print( anInt + " ");
	}
	System.out.println();
	
	/////////////////////////// 2. Vectors ///////////////////////////
	System.out.println("\nVectors:");
	Vector<String> stringVector = new Vector<String>();
	String[] stringArray = {"A", "B", "C"};
	stringVector = new Vector<String>(Arrays.asList(stringArray));
	System.out.println( stringVector.toString() );
	stringVector.add("D");
	System.out.println( stringVector.toString() );
	String vectorValue = stringVector.get(1);
	System.out.println(vectorValue) ; 
	String vectorValue2 = stringVector.elementAt(1);
	System.out.println(vectorValue2) ; 
	System.out.println(stringVector.size());
	stringVector.remove(1);
	System.out.println( stringVector.toString() );
	System.out.println( stringVector.size() );	
	stringVector.add(1, "B"); // (Index, element)
	System.out.println( stringVector.toString() );
	System.out.println( stringVector.size() );
	
	// Print Vector: Method 1
	Iterator it = stringVector.iterator();
	while(it.hasNext())
	{
	    System.out.print( it.next() + " ");
	}
	System.out.println();
	// Print Vector: Method 2
	for(int i=0; i < stringVector.size(); i++)
	{
	    System.out.print(stringVector.get(i) + " "); // stringVector[i] does not work
	}
	System.out.println();
	// Print vector: method 3
	System.out.println( stringVector.toString() );
	System.out.println();

	/////////////////////////// 3. Strings ///////////////////////////
	System.out.println("\nStrings:");
	String string1 = "Hello World";
	String string2 = new String("Hello World");
	System.out.println(string1);
	System.out.println(string2);
	// String equality
	System.out.println(string1.equals(string2) ); // true
	System.out.println(string1 == string2 ); // false
	// Retrieve elements, size
	System.out.println( string1.charAt(1) );
	System.out.println(string1.length());
	// Substrings
	System.out.println(string1.substring(1)); //beginIndex=1, inclusive. (Print string at index 1 onward.)
	System.out.println(string1.substring(1,4)); //beginIndex=1 inclusive, endIndex=4 exclusive (Print characters at indices [1,3].)
	
	// Print
	for(int i=0; i < string1.length(); i++) // length(), not length.
	{
	    System.out.print(string1.charAt(i) + " "); // string1.charAt(i), not string1[i]
	}
	System.out.println();
	// Print
	for(int i=0; i < string1.length(); i++) // length(), not length.
	{
	    // ASCII table: [A-Z] = [65, 90], [a-z] = [97, 122]
	    // http://www.asciitable.com/
	    System.out.print( (string1.charAt(i) - 'A') + " ");
	}
	// Output: 7 36 43 43 46 -33 22 46 49 43 35
	System.out.println();

	// String as character array
	char[] charArray = string1.toCharArray();
	// Print
	for(int i=0; i < charArray.length; i++) // length, not length().
	{
	    System.out.print(charArray[i] + " "); // charArray[i], not charArray.charAt(i)
	}
	System.out.println();
	System.out.println(charArray);
	System.out.println(Arrays.toString(charArray));
    }
}
