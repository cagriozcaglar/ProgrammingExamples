# From https://staffengprep.com/companies/openai/
# and https://aonecode.com/Interview-Question/Design-and-implement-Resumable-Iterator

'''
1. Resumable Iterators:
This is a test driven coding question. Interviewers ask to define an interface for resumable iterators. For eg:

class IteratorInterface:
    def __init__(self):
    def __iter__(self):
    def __next__(self):
    def get_state(self):
    def set_state(self, state):
Part A: Write a test class to test set_state and get_state functionality of this iterator interface.
For every step of iteration, capture the state and resume iterator using the captured state and validate the output
 A sample test case would look something like below:

def test_iterator(my_iter):
states = []
while my_iter.hasNext(): # Needs to handle when my_iter exhuasted
	states.append(my_iter.get_state())
	for state in states:
		test_iter = my_iter.set_state(state)
		elements = all elements from test_iter to the end
		assert elements == expected_elements

Part B: Implement IteratorInterface functions : next , get_state and set_state .
This shouldn’t take too long to implement. If using a typed language, get yourself
familiar with Generics as one will have to implement a generics interface.

Part C: Given an implementation of IteratorInterface to read from a single Json file
(lets call it JsonFileIterator which returns the next token in a single Json file.
Implement another iterator (lets call it MultipleJsonFileIterator ) which implements
an iterator over a set of files using SingleJsonFileIterator . These iterators should
support resumable capabilities so pay special attention to how set_state and get_state
functions get implemented. We will run the same tests written in Part A to test
MultipleJsonFileIterator.
'''
# Solution hint: https://www.google.com/search?q=A+common+openai+coding+question+involves+designing+and+implementing+a+resumbale+iterator

class IteratorInterface:
    def __init__(self):
        pass
    # Return an iterable object
    def __iter__(self):
        pass
    # Return next item
    def __next__(self):
        pass
    # Return a placeholder state for the current state of iterator
    def get_state(self):
        pass
    # Using a placeholder state, resume the iterator from the place where the state refers to
    def set_state(self, state):
        pass


class ResumableIterator:
    def __init__(self, data):
        self.data = data
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index < len(self.data):
            result = self.data[self.current_index]
            self.current_index += 1
            return result
        else:
            raise StopIteration

    def get_state(self):
        return self.current_index

    def set_state(self, index):
        if 0 <= index <= len(self.data):
            self.current_index = index
        else:
            raise ValueError("Invalid state")

# Example usage
data = [1, 2, 3, 4, 5]
iterator = ResumableIterator(data)

# Iterate partially
print(next(iterator))  # Output: 1
print(next(iterator))  # Output: 2

# Save the state
state = iterator.get_state() # state = 2

# Reset the iterator to a previous state
iterator.set_state(0) # Reset to index 0

# Resume from the saved state
print(next(iterator)) # Output: 1
print(next(iterator)) # Output: 2