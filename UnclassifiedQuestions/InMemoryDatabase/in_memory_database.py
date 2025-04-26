'''
Implement an in-memory database with insert, select, and conditional select methods, for a coding interview question.

Make sure conditional select supports conditions with >, <, = operators.

Answer is from ChatGPT:
https://chatgpt.com/g/g-iYSeH3EAI-website-generator/c/67f82e3a-8d1c-8000-9e96-457e155a46e8
'''

class InMemoryDB:
    def __init__(self):
        self.data = []
        self.columns = set()

    def insert(self, row):
        for col, val in row.items():
            if not isinstance(val, list) or not all(isinstance(item, dict) for item in val):
                raise ValueError("Each cell must be a list of dictionaries.")
        self.columns.update(row.keys())
        self.data.append(row)

    def select_all(self, select_columns=None):
        """
        Returns full rows or selected fields from specified columns.
        select_columns: dict like {'user': ['id', 'name'], 'details': ['age']}
        """
        if not select_columns:
            return self.data.copy()

        result = []
        for row in self.data:
            selected_row = {}
            for col, fields in select_columns.items():
                if col in row:
                    selected_row[col] = [
                        {k: v for k, v in entry.items() if k in fields}
                        for entry in row[col]
                    ]
            result.append(selected_row)
        return result

    def select_where(self, conditions, select_columns=None):
        """
        conditions: list of (column_name, key, operator, value)
        select_columns: same as select_all
        """
        def match(entry, key, op, val):
            if key not in entry:
                return False
            actual_value = entry[key]
            if op == '=':
                return actual_value == val
            elif op == '<':
                return actual_value < val
            elif op == '>':
                return actual_value > val
            else:
                raise ValueError(f"Unsupported operator: {op}")

        filtered = []
        for row in self.data:
            if all(
                col in row and any(match(entry, key, op, val) for entry in row[col])
                for col, key, op, val in conditions
            ):
                filtered.append(row)

        if not select_columns:
            return filtered

        # Project only selected fields
        result = []
        for row in filtered:
            selected_row = {}
            for col, fields in select_columns.items():
                if col in row:
                    selected_row[col] = [
                        {k: v for k, v in entry.items() if k in fields}
                        for entry in row[col]
                    ]
            result.append(selected_row)
        return result

# Example usage:
db = InMemoryDB()
db.insert({
    'user': [{'id': 1, 'name': 'Alice'}],
    'details': [{'age': 30, 'location': 'NY'}]
})
db.insert({
    'user': [{'id': 2, 'name': 'Bob'}],
    'details': [{'age': 25, 'location': 'LA'}]
})
db.insert({
    'user': [{'id': 3, 'name': 'Charlie'}, {'id': 4, 'name': 'Adam'}],
    'details': [{'age': 35, 'location': 'NY'}]
})

print("Select id, name from user:", db.select_where(
    conditions=[],
    select_columns={'user': ['id', 'name']}
))

print("Select id from user where age > 30:", db.select_where(
    conditions=[('details', 'age', '>', 30)],
    select_columns={'user': ['id']}
))

db.select_where(
    conditions=[('details', 'age', '>', 30)],
    select_columns={'user': ['id']}
)
