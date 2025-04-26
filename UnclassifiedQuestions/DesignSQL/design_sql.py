'''
Leetcode 2408: Design SQL

Design an SQL database to store the following information:

You are given two string arrays, names and columns, both of size n. The ith table is represented by the name names[i] and contains columns[i] number of columns.

You need to implement a class that supports the following operations:

1) Insert a row in a specific table with an id assigned using an auto-increment method, where the id of the first inserted row is 1, and the id of each new row inserted into the same table is one greater than the id of the last inserted row, even if the last row was removed.
2) Remove a row from a specific table. Removing a row does not affect the id of the next inserted row.
3) Select a specific cell from any table and return its value.
4) Export all rows from any table in csv format.

Implement the SQL class:
1) SQL(String[] names, int[] columns): Creates the n tables.
2) bool ins(String name, String[] row): Inserts row into the table name and returns true. If row.length
does not match the expected number of columns, or name is not a valid table, returns false without any insertion.
3) void rmv(String name, int rowId): Removes the row rowId from the table name. If name is not a valid table or
there is no row with id rowId, no removal is performed.
4) String sel(String name, int rowId, int columnId): Returns the value of the cell at the specified rowId and
columnId in the table name. If name is not a valid table, or the cell (rowId, columnId) is invalid, returns "<null>".
5) String[] exp(String name): Returns the rows present in the table name. If name is not a valid table, returns an
empty array. Each row is represented as a string, with each cell value (including the row's id) separated by a ",". 
'''

from typing import Dict, List
from collections import defaultdict

class SQL:

    def __init__(self, names: List[str], columns: List[int]):
        self.ids = {name: 1 for name in names}
        self.table_sizes = dict(zip(names, columns))
        self.tables: Dict[str, Dict[int, List[str]]] = {name: defaultdict(list) for name in names}

    def ins(self, name: str, row: List[str]) -> bool:
        if name not in self.table_sizes or len(row) != self.table_sizes[name]:
            return False

        self.tables[name][self.ids[name]] = row
        self.ids[name] += 1
        return True

    def rmv(self, name: str, rowId: int) -> None:
        if name in self.tables and rowId in self.tables[name]:
            del self.tables[name][rowId]

    def sel(self, name: str, rowId: int, columnId: int) -> str:
        if name not in self.tables or rowId not in self.tables[name]:
            return "<null>"
        return self.tables[name][rowId][columnId - 1]

    def exp(self, name: str) -> List[str]:
        csv = []
        for index, row in self.tables[name].items():
            csv.append(",".join([str(index)] + row))
        return csv
