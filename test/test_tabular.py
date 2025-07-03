import numpy as np
import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.numpy
from hipy.lib._tabular import table, column

@hipy.compiled_function
def fn_column_creation():
    c1 = column([1, 2, 3])
    c2 = column(["a", "b", "c"])
    c3 = column([True, False, True])
    c4 = column([[1, 2], [3, 4], [5, 6]])
    print(c1)
    print(c2)
    print(c3)
    print(c4)

def test_column_creation():
    check_prints(fn_column_creation, """
[
  [
    1,
    2,
    3
  ]
]
[
  [
    "a",
    "b",
    "c"
  ]
]
[
  [
    true,
    false,
    true
  ]
]
[
  [
    [
      1,
      2
    ],
    [
      3,
      4
    ],
    [
      5,
      6
    ]
  ]
]

""")

@hipy.compiled_function
def fn_table_creation():
    c1 = column([1, 2, 3])
    c2 = column(["a", "b", "c"])
    t = table({"a": c1, "b": c2})
    print(t)

def test_table_creation():
    check_prints(fn_table_creation, """
pyarrow.Table
a: int64
b: string
----
a: [[1,2,3]]
b: [["a","b","c"]]
""")



@hipy.compiled_function
def fn_column_ops():
    c = column([1, 2, 3])
    c2 = column([10, 100, 1000])
    print(c)
    print(len(c))
    print(c.apply(lambda x: x * x))
    print(c.element_wise(c2, lambda x, y: x * y))
    print(c.filter_by_column(column([True, False, True])))
    print(c.aggregate(0, lambda x, y: x + y, lambda x, y: x + y))
    print(column([1, 1, 2, 2, 3, 3]).unique())
    print(column([1, 2, 3]).isin(column([1, 3])))
    print(column.sequential(10))

def test_column_ops():
    check_prints(fn_column_ops, """

[
  [
    1,
    2,
    3
  ]
]
3
[
  [
    1,
    4,
    9
  ]
]
[
  [
    10,
    200,
    3000
  ]
]
[
  [
    1,
    3
  ]
]
6
[
  [
    1,
    2,
    3
  ]
]
[
  [
    true,
    false,
    true
  ]
]
[
  [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9
  ]
]
""")

@hipy.compiled_function
def fn_table_ops():
    t = table({"a": column([2, 1, 3,5,2]), "b": column(["a", "b", "c","d","e"]), "c": column([True, False, True,True, False])})
    print(t)

    print(t.sort(["a","b"], [True,False]))
    print(t.filter_by_column(column([True, False, True,False,True])))
    print(t.get_column("a"))
    print(t.select_columns(["a", "b"]))
    print(t.get_slice(slice(1,3)))
    print(t.get_slice(slice(-3,-1)))
    print(len(t))
    print(t.set_column("b", column(["d", "e", "f","g","h"]))) # replace existing column
    print(t.set_column("d", column([1, 2, 3,4,5]))) # add new column
    print(t.set_column("a", column(["d", "e", "f","g","h"]))) # replace existing column with different type
    print(t.apply_row_wise(lambda x: str(x["a"]) + x["b"]))

def test_table_ops():
    check_prints(fn_table_ops, """
pyarrow.Table
a: int64
b: string
c: bool
----
a: [[2,1,3,5,2]]
b: [["a","b","c","d","e"]]
c: [[true,false,true,true,false]]
pyarrow.Table
a: int64
b: string
c: bool
----
a: [[1,2,2,3,5]]
b: [["b","e","a","c","d"]]
c: [[false,false,true,true,true]]
pyarrow.Table
a: int64
b: string
c: bool
----
a: [[2,3,2]]
b: [["a","c","e"]]
c: [[true,true,false]]
[
  [
    2,
    1,
    3,
    5,
    2
  ]
]
pyarrow.Table
a: int64
b: string
----
a: [[2,1,3,5,2]]
b: [["a","b","c","d","e"]]
pyarrow.Table
a: int64
b: string
c: bool
----
a: [[1,3]]
b: [["b","c"]]
c: [[false,true]]
pyarrow.Table
a: int64
b: string
c: bool
----
a: [[3,5]]
b: [["c","d"]]
c: [[true,true]]
5
pyarrow.Table
a: int64
b: string
c: bool
----
a: [[2,1,3,5,2]]
b: [["d","e","f","g","h"]]
c: [[true,false,true,true,false]]
pyarrow.Table
a: int64
b: string
c: bool
d: int64
----
a: [[2,1,3,5,2]]
b: [["a","b","c","d","e"]]
c: [[true,false,true,true,false]]
d: [[1,2,3,4,5]]
pyarrow.Table
a: string
b: string
c: bool
----
a: [["d","e","f","g","h"]]
b: [["a","b","c","d","e"]]
c: [[true,false,true,true,false]]
[
  [
    "2a",
    "1b",
    "3c",
    "5d",
    "2e"
  ]
]""")


@hipy.compiled_function
def fn_get_by_index():
    c = column([1, 2, 3])
    print(c.get_by_index(1))
    print(c.get_by_index(2))

def test_get_by_index():
    check_prints(fn_get_by_index, """
2
3
""")


@hipy.compiled_function
def fn_column_from_array():
    c = column(np.ones(10))
    print(c)

def test_column_from_array():
    check_prints(fn_column_from_array, """
[
  [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1
  ]
]
""")