"""Generator expressions `(expr for target in iter [if cond ...])`.

The rewriter lowers these in compiler.py:196, packaging the element
expression into helper closures and a type-inference callback
(context.infer_return_type). Previously no test exercised this path.
"""
import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant


@hipy.compiled_function
def fn_genexp_sum():
    print(sum(x * 2 for x in range(5)))


def test_genexp_sum():
    check_prints(fn_genexp_sum, "20")


@hipy.compiled_function
def fn_genexp_filter():
    # generator expression with a guard — exercises wrap_ifs in the
    # rewriter (compiler.py:204-211).
    print(sum(x for x in range(10) if x % 2 == 0))


def test_genexp_filter():
    check_prints(fn_genexp_filter, "20")


@hipy.compiled_function
def fn_genexp_closed_over():
    # free variable from the enclosing scope — exercises packed_vals.
    base = not_constant(100)
    print(sum(base + x for x in range(3)))


def test_genexp_closed_over():
    check_prints(fn_genexp_closed_over, "303")
