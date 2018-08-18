/**
 HashTable implementation
 */

// Implement using an array of LinkedLists

class Entry{
    String key;
    String value;

    public Entry(String key, String value) {
        this.key = key;
        this.value = value;
    }
}

class HashtableLinkedHashing {
    private LinkedList[] list;
    private int initialSize;
    private double factor;

    public HashtableLinkedHashing(int initialSize, double factor) {
        this.initiallSize = initialSize;
        this.factor = factor;
    }

    public void put(String key, String value) {
        // Error checks
        if(key == null){
            throw new IllegalArgumentException();
        }

        long hashValue = hash(key);
        int index = hashValue%size;
        List list = lists[index];
        Entry entry = new Entry(key, value);
        if(list != null){
            list = new LinkedList();
            lists[i] = list;
        }

        // Check if value exists, insert
        for(Entry entry : list) {

        }
    }

    public String get(String key){
        if(key == null) {
            throw new IllegalArgumentException();
        }
        int index = getIndex(key);
        List list = lists[index];
        if(list == null){
            return null;
        }

        for(Entry e : list) {
            if(e.key.equals(key)) {
                return e.value;
            }
        }
        return null;
    }

    public long getIndex(String key) {
        long hashValue = hash(key);
        int index = hashValue % size;
        return index;
    }

    // Override hash function of the class
    // Commonly used hash functions
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ( (key==null) ? 0: key.hashCode() );
        return result;
    }

    public static void main(String[] args){

    }

}


/**
 // package whatever; // don't place package name!

 import java.io.*;


 class HashtableOpenAddressing {

 public static void main (String[] args) {
 System.out.println("Hello Java");
 }
 }

 class Entry{
 String key;
 String value;

 public Entry(String key, String value){
 this.key = key;
 this.value = value;
 }

 }

 class HashtableLinkedHashing {
 private LinkedList[] lists;
 private int size;
 private double factor;

 public HashtableLinkedHashing(int initialSize,double factor)
 {
 this.size = initialSize;
 this.factor = factor;
 }


 public void put(String key, String value){

 if (key == null){
 throw new IllegalArgumentException();
 }

 int index = getIndex(key);
 List list = lists[index];
 Entry e = new Entry(key,value);
 if (list == null){
 list = new LinkedList();
 lists[i]=list;
 }

 for(Entry e: list){
 if (e.key.equals(key)){
 e.value = value;
 }
 }

 list.add(e);

 }


 public void get(String key){
 if (key == null){
 throw new IllegalArgumentException();
 }

 int index = getIndex(key);
 List list = lists[index];
 if (list == null){
 return null;
 }

 for(Entry e : list){
 if (e.key.equals(key)){
 return e.value;
 }
 }

 return null;

 }

 private long getIndex(String key ){
 long hashValue = hash(key);
 int index = hashValue%size;
 return index;
 }



 @Override
 public int hashCode() {
 final int prime = 31;
 int result = 1;
 result = prime * result + Arrays.hashCode(heap);
 result = prime * result + initialSize;
 result = prime * result + last;
 result = prime * result + (minHeap ? 1231 : 1237);
 return result;
 }

 private long hash(String key){
 final int prime = 31;
 int result = 1;
 result = prime * result + ((key == null) ? 0 : key.hashCode());
 return result;
 }


 public static void main (String[] args) {
 System.out.println("Hello Java");
 }
 }


*/