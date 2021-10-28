# Given a log file of function running times in the following format:
### [functionName],[timeStamp],[beginOrEnd]
# Return the exclusive running time of each function in the log file
""" Example input:
f1      50.0    b
f2      80.0    b
f3      120.0   b
f2      200.0   e
f1      250.0   e
f4      290.0   b
f3      400.0   e
f4      500.0   e
"""

""" Example output:
f1   30.0
f2   40.0
f3   170.0
f4   210.0
"""

""" Calculations using the program below
f1: 30
f2: 40
f3: 80 + 50 + 40 = 170
f4: 110 + 100 = 210
"""

import sys
from collections import defaultdict

# Parameters:
## [logFileName]: name of the function log file
scriptName, logFileName = sys.argv

# callStack: to keep track of which function is currently running.
# Python doesn't have a built-in stack data structure, so we use a list as stack.
# Operations: append() for push, pop() for pop operations.
callStack = []
callStackTemp = []
# runningTimeMap: HashMap from function name its total exclusive running time.
runningTimeMap = defaultdict(int)

#logFileName="functionLog.txt"
lines = open(logFileName,'r')
timePrev = None
timeCur = None
for line in lines:
    #records = line.strip('\n').split('\t')
    #print(('\t'.join(records)))
    # Parse records
    [functionName, timeStampString, beginOrEnd] = line.strip('\n').split('\t')
    timeCur = float(timeStampString)
    print(functionName + ', ' + timeStampString + ", " + beginOrEnd) 
    # If timePrev is None, then this is the first line of the function log.
    if timePrev is None:
        # If a function begins as the first element, push function to callStack.
        if beginOrEnd == 'b':
            callStack.append(functionName)
            print("Append " + functionName + " to call stack.")
        # If a function ends, given that there is no timePrev, return error.
        elif beginOrEnd == 'e':
            sys.exit('Error: Function call stack is empty, no function to remove from call stack.')
    # If timePrev is assigned, e.g. this is not the first line of the function log
    elif timePrev is not None:
        # Get top element, this is the function which has been running from timePrev to timeCur.
        # Add (timeCur-timePrev) to the exclusive running time of this function.
        topElement = callStack[-1]
        runningTimeMap[topElement] = runningTimeMap[topElement] + (timeCur - timePrev)
        # If the new function is beginning, push it to call stack.
        if beginOrEnd == 'b':
            callStack.append(functionName)
            print("Append " + functionName + " to call stack.")
        # If the new function is ending, pop it from the stack
        elif beginOrEnd == 'e':
            while len(callStack) > 0 and functionName != callStack[-1]:
                callStackTemp.append(callStack.pop())
                print("Append " + callStackTemp[-1] + " from call stack to temporary call stack.")
            if len(callStack) == 0:
                sys.exit('Error: Function does not exist in the call stack.')
            print("Pop element from call stack, append elements in temporary call stack to actual call stack.")
            callStack.pop()
            for element in callStackTemp:
                callStack.append(callStackTemp.pop())
    # For all cases, update assign timePrev to timeCur
    timePrev = timeCur
    print("Call Stack:" + str(callStack))

for functionName,exclusiveRunningTime in runningTimeMap.items():
    print(functionName + "\t" + str(exclusiveRunningTime))
