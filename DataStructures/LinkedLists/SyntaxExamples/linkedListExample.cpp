#include <iostream>
using namespace std;

class LinkedList
{
  struct Node
  {
    int x;
    Node* next;
  }; // Note the semicolon (;) at the end of the struct.

public:
  LinkedList()
  {
    head = NULL; // Set head to NULL
  }

  void addToHead(int val)
  {
    Node* n = new Node();  // Create new node
    n->x = val;            // Set the value of new node
    n->next = head;        // Set next of the new node as the former head
    head = n;              // Set head to be the new node
  }

  int deleteFromHead()
  {
    Node* n = head;
    int ret = n->x;
    head = head->next;
    delete n;
    return ret;
  }

private:
  Node* head;

}; // Note the semicolon at the end of class.

int main()
{
  LinkedList linkedList;
  linkedList.addToHead(1);
  linkedList.addToHead(2);
  linkedList.addToHead(3);
  cout << linkedList.deleteFromHead() << endl;
  cout << linkedList.deleteFromHead() << endl;
  cout << linkedList.deleteFromHead() << endl;
}
