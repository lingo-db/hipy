import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant


# `return` inside loops (compiler.rewrite_loop_return) and break/continue in
# nested if/elif chains. Loop bounds are made non-constant so that the loops
# are really staged as loops (not unrolled).

@hipy.compiled_function
def find_first_square_above(x):
    for i in range(not_constant(100)):
        if i * i > x:
            return i
    return -1


@hipy.compiled_function
def fn_return_in_for():
    print(find_first_square_above(not_constant(50)))
    print(find_first_square_above(not_constant(20000)))


def test_return_in_for():
    check_prints(fn_return_in_for, """8
-1""")


@hipy.compiled_function
def count_up(x):
    i = 0
    while True:
        i += 1
        if i > x:
            return i * 10


@hipy.compiled_function
def fn_return_in_while_true():
    # a `while True` loop without break can only be left by a return
    print(count_up(not_constant(5)))


def test_return_in_while_true():
    check_prints(fn_return_in_while_true, """60""")


@hipy.compiled_function
def find_product(x):
    for i in range(not_constant(10)):
        j = 0
        while j < 10:
            if i * j == x:
                return "found " + str(i) + "*" + str(j)
            j += 1
    return "none"


@hipy.compiled_function
def fn_return_in_nested_loops():
    print(find_product(not_constant(12)))
    print(find_product(not_constant(97)))


def test_return_in_nested_loops():
    check_prints(fn_return_in_nested_loops, """found 2*6
none""")


@hipy.compiled_function
def fn_break_while_true():
    # the loop must stop after the break, not only skip the rest of the body
    i = 0
    x = not_constant(5)
    while True:
        i += 1
        if i > x:
            break
    print(i)


def test_break_while_true():
    check_prints(fn_break_while_true, """6""")


@hipy.compiled_function
def fn_continue_in_elif():
    acc = 0
    for i in range(not_constant(7)):
        if i % 3 == 0:
            acc += 1
        elif i % 3 == 1:
            continue
        else:
            acc += 100
        acc += 10
    print(acc)


def test_continue_in_elif():
    check_prints(fn_continue_in_elif, """253""")


@hipy.compiled_function
def fn_break_in_elif():
    acc = 0
    i = 0
    x = not_constant(9)
    while i < 100:
        if i == x:
            break
        elif i % 2 == 0:
            acc += i
        else:
            acc -= 1
        i += 1
    print(acc, i)


def test_break_in_elif():
    check_prints(fn_break_in_elif, """16 9""")
