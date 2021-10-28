#include<iostream>   // std:cout
#include<stack>      // std:stack
#include<queue>      // std:queue

using namespace std;

int main()
{
  // Stack
  cout << "Stacks:" << endl;
  stack<int> intStack;
  // Push: push()
  intStack.push(1);
  intStack.push(2);
  intStack.push(3);

  // Size: size()
  cout << intStack.size() << endl;
  // Empty: empty()
  cout << intStack.empty() << endl;

  // Pop: pop()
  intStack.pop();
  // Top: top()
  cout << intStack.top() << endl;

  intStack.push(3);
  // Print stack
  for(stack<int> tempStack = intStack; !tempStack.empty(); tempStack.pop())
  {
    cout << tempStack.top() << " ";  
  }
  cout << endl << endl;

  // Queue
  cout << "Queues:" << endl;
  queue<int> intQueue;
  // Push element: push()
  intQueue.push(1);
  intQueue.push(2);
  intQueue.push(3); 
  // Print queue
  for(queue<int> tempQueue = intQueue; !tempQueue.empty(); tempQueue.pop())
  {
    cout << tempQueue.front() << " ";  
  }
  cout << endl;
  /// Output: 1 2 3

  // Remove element: pop()
  intQueue.pop();
  for(queue<int> tempQueue = intQueue; !tempQueue.empty(); tempQueue.pop())
  {
    cout << tempQueue.front() << " ";  
  }
  cout << endl;
  /// Output: 2 3
  intQueue.push(1);
  for(queue<int> tempQueue = intQueue; !tempQueue.empty(); tempQueue.pop())
  {
    cout << tempQueue.front() << " ";  
  }
  cout << endl;
  /// Output: 2 3 1

  // Get first element: front()
  cout << intQueue.front() << endl; // 2
  // get last element: back()
  cout << intQueue.back() << endl;  // 1

  // Check if empty: empty()        
  cout << intQueue.empty() << endl; // 0
  // Get size: size()
  cout << intQueue.size() << endl;  // 3

  // Emplace: emplace(). Not supported in C++ anymore.
  // intQueue.emplace(4);
}
