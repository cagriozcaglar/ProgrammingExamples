/**
 Implement LinkedList operations with a generic type
 */

class Node<T>{
    public Node next;
    public T value;
    public Node(T value){
        this.value = value;
    }

    public void add(T value){
        if(value == null){
            throw new IllegalArgumentException();
        }
        if(head == null){
            Node<T> newNode = new Node(value);
            head = newNode;
            tail = newNode;
            return;
        }
        Node<T> newNode = new Node(value);
        newNode.next = head;
        head = newNode;
    }

    public void addToTail(T value){
        if(value == null){
            throw new IllegalArgumentException();
        }
        if(tail == null){
            Node<T> newNode = new Node(value);
            head = newNode;
            tail = head;
            return;
        }
        Node<T> newNode = new Node(value);
        tail.next = newNode;
        tail = newNode;
    }

    public void insertAt(int i, T value){
        // Error checks
        if(value == null || i < 0){
            throw new IllegalArgumentException();
        }

        Node<T> newNode = new Node(value);

        // Iterate and insert @i in the middle
        if(i == 0){
            Node n = new Node(value);
            n.next = head;
            head = n;
            return;
        }

        Node prev = null;
        Node<T> current = head;
        int pos = 0;
        for(pos = 0; pos < i && current != null; pos++){
            prev = current;
            current = current.next;
        }

        Node newNode = new Node(v);
        Node t = prev.next;
        prev.next = new Node(v);
        newNode.next = t;

        Node<T> prev = new Node(-1);
        prev.next = head;
        Node<T> current = head;

    }


    public T remove remove(int at){

    }

    public void remove(Object t){

    }

    public void printList(){

    }
}


// import java.io.*;
// class Node<T>{
// public Node next;
// public T value;
// public Node(T value ){
// this.value = value;
// }
// }
//
// class LinkedList<T> {
//
// private Node<T> head;
// private Node<T> tail;
//
// O(1)  -- addition
// O(N)-- ?
/*
public void add(T value){
    if (value == null){
        throw new IllegalArgumentException();
    }

    if (head == null){
        Node<T> newNode = new Node(value);
        head = newNode;
        tail = newNode;
        return;
    }
    Node<T> newNode = new Node(value);
    newNode.next = head;
    head = newNode;
}

     // value = null -
     // 0; 1
     // H T
     // 1; [1,2]
     //
    public void addToTail(T value){
        if (value == null){
            throw new IllegalArgumentException();
        }

        if (tail == null){
            Node<T> newNode = new Node(value);
            head = newNode;
            tail = head;
            return;
        }

        Node<T> newNode = new Node(value);
        tail.next = newNode;
        tail = newNode;
    }


    public void insertAt(int i, T value ){

        if(value == null)
        {
            throw new IllegalArgumentException();
        }
        if(i < 0)
        {
            throw new IllegalArgumentException();
        }

        Node<T> newNode = new Node(value);

        //iterate and insert @ i in the middle
        if (i == 0 ){
            Node n = new Node(value);
            n.next = head;
            head = n;
            return;
        }

        //find pos
        Node prev = null;
        Node<T> current = head;
        int pos=0;
        for (pos=0;pos<i&& current !=null;pos++){
            prev = current;
            current = current.next;
        }


        //insert node
        if (pos == i){
            Node newNode = new  Node(v);
            Node t = prev.next;
            prev.next = new  Node(v);
            newNode.next = t;
        }else{
            throw an execptoin
        }

    }

    public T remove(int at){
    }

    public void remove(Object t){
    }

    public void printList(){
    }


    public static void main (String[] args) {
        LinkedList<Integer> list = new LinkedList<Integer>();
        list.add(1);
        list.add(2);
        list.printList();
        list.insertAt(2,3);
        list.printList();

    }
}

*/
