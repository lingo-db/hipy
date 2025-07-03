import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant


@hipy.compiled_function
def fn_range_construction_(r):
    print(r.start)
    print(r.stop)
    print(r.step)
    print(r)


@hipy.compiled_function
def fn_range_construction():
    fn_range_construction_(range(1))
    fn_range_construction_(range(1, 2))
    fn_range_construction_(range(1, 2, 3))
    fn_range_construction_(not_constant(range(1)))
    fn_range_construction_(not_constant(range(1, 2)))
    fn_range_construction_(not_constant(range(1, 2, 3)))


def test_range_construction():
    check_prints(fn_range_construction, """
0
1
1
range(0, 1)
1
2
1
range(1, 2)
1
2
3
range(1, 2, 3)
0
1
1
range(0, 1)
1
2
1
range(1, 2)
1
2
3
range(1, 2, 3)""")


@hipy.compiled_function
def range_iter_(r):
    for i in r:
        print(i)

@hipy.compiled_function
def range_iter():
    range_iter_(range(5))
    range_iter_(range(1, 5))
    range_iter_(range(1, 5, 2))
    range_iter_(not_constant(range(1)))
    range_iter_(not_constant(range(1, 5)))
    range_iter_(not_constant(range(1, 5, 2)))


def test_range_iter():
    check_prints(range_iter, """
0
1
2
3
4
1
2
3
4
1
3
0
1
2
3
4
1
3
    """)
