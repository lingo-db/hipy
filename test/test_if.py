import pytest

import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.compiler
"""
variables only defined in one if branch can not be used afterwards
"""


@hipy.compiled_function
def fn_if_not_available():
    a = not_constant(True)
    if a:
        b = False
    print(b)


def test_if_not_available():
    with pytest.raises(NameError) as e:
        hipy.compiler.compile(fn_if_not_available, fallback=False)
    assert "name 'b' is not defined" in e.value.args[0]
    # as comp_error:
    #    dbpy.translator.compile(fn_if_not_available, forward_error=True, printErrors=False)

    #assert "variable `b` is not available depending on the control flow" in comp_error.value.get_first().message


"""
variables only defined in one if branch can not be used afterwards
"""


@hipy.compiled_function
def fn_if_tmp_vars_ok():
    a: bool = True  # type annotation is required to avoid compile time computation
    if a:
        x = 0
        b = False
    else:
        y = 1
        b = True
    print(b)


def test_if_tmp_vars_ok():
    check_prints(fn_if_tmp_vars_ok, """False""")


"""
the type of variables must be the same after both branches
"""


@hipy.compiled_function
def fn_if_different_types():
    a= not_constant(True)
    x = 1
    if a:
        x = 1.0
    print(x)


def test_if_different_types():
    check_prints(fn_if_different_types,"""1.0""" )


"""
test for correct behavior of if statement
"""


@hipy.compiled_function
def fn_if():
    t: bool = True  # type annotation is required to avoid compile time computation
    f: bool = False  # type annotation is required to avoid compile time computation

    if t:
        print("t")
        res = t
    else:
        print("f")
        res = f
    print(res)
    if f:
        print("t")
        res = t
    else:
        print("f")
        res = f
    print(res)


def test_if():
    check_prints(fn_if, """
t
True
f
False
""")


"""
test for correct behavior of if expr
"""


@hipy.compiled_function
def fn_if_expr():
    t = not_constant(True)
    f = not_constant(False)
    print(True if t else False)
    print(True if f else False)
    print(True if True else False)
    print(True if False else False)


def test_if_expr():
    check_prints(fn_if_expr, """
True
False
True
False
""")


@hipy.compiled_function
def fn_if_return_(b):
    if b:
        return True
    else:
        return False

@hipy.compiled_function
def fn_if_return():
    print(fn_if_return_(True))
    print(fn_if_return_(False))
    print(fn_if_return_(not_constant(True)))
    print(fn_if_return_(not_constant(False)))

def test_if_return():
    check_prints(fn_if_return, """
True
False
True
False
""")

@hipy.compiled_function
def fn_if_return_one_branch_(b):
    if b:
        return True
    else:
        pass
    return False

@hipy.compiled_function
def fn_if_return_one_branch_const():
    t_res=fn_if_return_one_branch_(True)
    f_res=fn_if_return_one_branch_(False)
    print(t_res)
    print(f_res)
    print("native?",intrinsics.isa(t_res, bool))
    print("native?",intrinsics.isa(f_res, bool))

def test_if_return_one_branch_const():
    check_prints(fn_if_return_one_branch_const, """
True
False
native? True
native? True
""")

@hipy.compiled_function
def fn_if_return_one_branch():
    t_res=fn_if_return_one_branch_(not_constant(True))
    f_res=fn_if_return_one_branch_(not_constant(False))
    print(t_res)
    print(f_res)
    print("native?",intrinsics.isa(t_res, bool))
    print("native?",intrinsics.isa(f_res, bool))
def test_if_return_one_branch():
    with pytest.raises(NotImplementedError) as e:
        hipy.compiler.compile(fn_if_return_one_branch)
    assert "early return in only one branch" in e.value.args[0]

    check_prints(fn_if_return_one_branch, """
True
False
native? False
native? False
    """,fallback=True)

@hipy.compiled_function
def fn_if_return_one_branch2_(b):
    if b:
        return True
    return False

@hipy.compiled_function
def fn_if_return_one_branch2():
    t_res=fn_if_return_one_branch2_(not_constant(True))
    f_res=fn_if_return_one_branch2_(not_constant(False))
    print(t_res)
    print(f_res)
    print("native?",intrinsics.isa(t_res, bool))
    print("native?",intrinsics.isa(f_res, bool))

def test_if_return_one_branch2():
    check_prints(fn_if_return_one_branch2, """
True
False
native? True
native? True
""")

@hipy.compiled_function
def fn_if_changed_values():
    l=[[]]
    if not_constant(True):
        l[0].append(10)
    print(l[0])
def test_if_changed_values():
    check_prints(fn_if_changed_values, """
[10]
""")


@hipy.compiled_function
def fn_if_nested_changed_values():
    x = None
    if True:
        if True:
            x = 1
    print(x)

def test_if_nested_changed_values():
    check_prints(fn_if_nested_changed_values, """
1
""")


@hipy.compiled_function
def fn_test_if_nested_changed_values():
    x = 0
    if True:
        if False:
            x = 1
    print(x)

def test_test_if_nested_changed_values():
    check_prints(fn_test_if_nested_changed_values, """
0
""")