class Interval:
    def __init__(self,start,end):
        self.start = start
        self.end = end
    def getStart(self):
        return self.start
    def getEnd(self):
        return self.end
    def print(self):
        print("(" + str(self.start) + ", " + str(self.end) + ")")

if __name__ == '__main__':
    int1 = Interval(1.2, 4.5);
    int2 = Interval(2.3, 5.6)
    int1.print()
