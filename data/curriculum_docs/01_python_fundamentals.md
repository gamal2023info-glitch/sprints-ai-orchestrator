# Python Fundamentals (PLACEHOLDER CONTENT - replace with real Sprints curriculum)

## Lists vs tuples
A list is mutable: you can append, remove and reassign items. A tuple is immutable:
once created it cannot change, which makes it hashable (usable as a dictionary key) and
slightly faster. Use lists for collections that change and tuples for fixed records,
such as coordinates (30.04, 31.23).

## Functions and docstrings
Define functions with `def`. A docstring in triple quotes directly under the signature
documents purpose, arguments and return value. Sprints graders award points for clear
docstrings and for functions that return values instead of only printing them.

## Error handling
Use try/except to handle expected failures. Catch specific exceptions such as
ValueError or ZeroDivisionError, never a bare except.
