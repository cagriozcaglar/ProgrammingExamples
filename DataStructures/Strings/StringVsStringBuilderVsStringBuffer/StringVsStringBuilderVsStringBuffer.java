/**
 * Comparison of String vs. StringBuilder vs. StringBuffer:
 * 1) String object is immutable in Java, but StringBuilder and StringBuffer are mutable objects.
 * 2) StringBuffer is synchronized, while StringBuilder is not synchronized, which makes StringBuilder faster than StringBuffer.
 * 3) Use String if you require immutability.
 *    Use StringBuffer if you need mutability and thread-safety.
 *    Use StringBuilder if you need mutability without thread-safety (which is faster).
 */
public class StringVsStringBuilderVsStringBuffer {

    public static void main(String[] args){
        // Timers for profiling
        long startTime;
        long endTime;

        // Instances of String, StringBuilder, StringBuffer
        String str = "TestString";
        StringBuilder sbdr = new StringBuilder("TestStringBuilder");
        StringBuffer sbfr = new StringBuffer("TestStringBuffer");

        /**
         * Print String, StringBuilder, StringBuffer
         */
        System.out.println("Printing String, StringBuilder, StringBuffer");
        System.out.println(str);
        System.out.println(sbdr);
        System.out.println(sbfr);

        /**
         * Test mutability of String, StringBuilder, StringBuffer
         * Profiling: Concatenation of String, StringBuilder, StringBuffer
         * Result:
         * 1) StringBuilder (7257 ns)
         * 2) StringBuffer (11956 ns)
         * 3) String (23896 ns)
         */
        System.out.println("\nTest mutability of String, StringBuilder, StringBuffer");
        // String: Immutable
        startTime = System.nanoTime();
        concatString(str);
        endTime =  System.nanoTime();
        System.out.println("Time for String concatenation: " + (endTime - startTime));
        // StringBuilder: Mutable
        startTime = System.nanoTime();
        concatStringBuilder(sbdr);
        endTime =  System.nanoTime();
        System.out.println("Time for StringBuilder concatenation: " + (endTime - startTime));
        // StringBuffer: Mutable
        startTime = System.nanoTime();
        concatStringBuffer(sbfr);
        endTime =  System.nanoTime();
        System.out.println("Time for StringBuffer concatenation: " + (endTime - startTime));
        // After concanetaion: Print String, StringBuilder, StringBuffer
        System.out.println("String after concatenation: " + str);
        System.out.println("StringBuilder after concatenation: " + sbdr);
        System.out.println("StringBuffer after concatenation: " + sbfr);

        /**
         * Conversions among String, StringBuilder, StringBuffer
         */
        System.out.println("\nConversions among String, StringBuilder, StringBuffer");
        // 1. Conversion from String to StringBuilder and StringBuffer
        // Pass String to StringBuilder / StringBuffer constructor
        System.out.println("Conversion from String to StringBuilder and StringBuffer");
        String strToConvert = "TestString";
        StringBuilder sbdrToConvert = new StringBuilder(strToConvert);
        StringBuffer sbfrToConvert = new StringBuffer(strToConvert);
        System.out.println("String to convert: " + strToConvert);
        System.out.println("StringBuilder after conversion from String: " + sbdrToConvert);
        System.out.println("StringBuffer after conversion from String: " + sbfrToConvert);

        // 2. Conversion from StringBuilder and StringBuffer to String
        // Use toString() method
        System.out.println("\nConversion from StringBuilder and StringBuffer to String");
        String stringConvertedFromStringBuilder = sbdrToConvert.toString();
        String stringConvertedFromStringBuffer = sbfrToConvert.toString();
        System.out.println("String converted from StringBuilder: " + stringConvertedFromStringBuilder);
        System.out.println("String converted from StringBuffer: " + stringConvertedFromStringBuffer);

        // 3. Conversion among StringBuilder and StringBuffer
        // Convert to String first, than to other class.
        System.out.println("\nConversion among StringBuilder and StringBuffer");
        // 3.1. Conversion From StringBuilder to StringBuffer
        StringBuilder sbdrToConvertToStringBuffer = new StringBuilder("StringBuilder to convert to StringBuffer");
        StringBuffer sfdrConvertedFromStringBuilder = new StringBuffer(sbdrToConvertToStringBuffer.toString());
        System.out.println("StringBuilder converted to StringBuffer: " + sfdrConvertedFromStringBuilder);
        // 3.2. From StringBuffer to StringBuilder
        StringBuffer sfdrToConvertToStringBuilder = new StringBuffer("StringBuffer to convert to StringBuilder");
        StringBuilder sbdrConvertedFromStringBuffer = new StringBuilder(sfdrToConvertToStringBuilder.toString());
        System.out.println("StringBuffer converted to StringBuilder: " + sbdrConvertedFromStringBuffer);

        /**
         * Profiling: Join words with String vs. StringBuilder vs. StringBuffer
         * Result:
         * 1) String        (17646 ns)
         * 2) StringBuilder (6939 ns)
         * 3) StringBuffer  (12425 ns)
         */
        System.out.println("\nProfiling: Join words with String vs. StringBuilder vs. StringBuffer");
        String[] words = {"How", " are", " you", " doing?"};

        startTime = System.nanoTime();
        String joinedWordsWithString = joinWordsWithString(words);
        endTime =  System.nanoTime();
        System.out.println("Time for joinedWordsWithString: " + (endTime - startTime));

        startTime = System.nanoTime();
        String joinedWordsWithStringBuilder = joinWordsWithStringBuilder(words);
        endTime =  System.nanoTime();
        System.out.println("Time for joinedWordsWithStringBuilder: " + (endTime - startTime));

        startTime = System.nanoTime();
        String joinedWordsWithStringBuffer = joinWordsWithStringBuffer(words);
        endTime =  System.nanoTime();
        System.out.println("Time for joinedWordsWithStringBuffer: " + (endTime - startTime));

        System.out.println("joinedWordsWithString: " + joinedWordsWithString);
        System.out.println("joinedWordsWithStringBuilder: " + joinedWordsWithStringBuilder);
        System.out.println("joinedWordsWithStringBuffer: " + joinedWordsWithStringBuffer);
    }

    /**
     * String concatenation using String vs. StringBuilder vs StringBuffer
     */
    // Concatenation to String
    public static void concatString(String str){
        str = str + " concatString";
    }

    // Concatenation to StringBuilder
    public static void concatStringBuilder(StringBuilder sbdr){
        sbdr.append(" concatStringBuilder");
    }

    // Concatenation to StringBuffer
    public static void concatStringBuffer(StringBuffer sbfr){
        sbfr.append(" concatStringBuffer");
    }

    /**
     * Join words using String vs. StringBuilder vs. StringBuffer
     */
    // Join words using String
    static String joinWordsWithString(String[] words) {
        String sentence = "";
        for (String w : words) {
            sentence = sentence + w;
        }
        return sentence;
    }

    // Join words using StringBuilder
    static String joinWordsWithStringBuilder(String[] words) {
        StringBuilder sentence = new StringBuilder();
        for (String w : words) {
            sentence.append(w);
        }
        return sentence.toString();
    }

    // Join words using StringBuffer
    static String joinWordsWithStringBuffer(String[] words) {
        StringBuffer sentence = new StringBuffer();
        for (String w : words) {
            sentence.append(w);
        }
        return sentence.toString();
    }
}