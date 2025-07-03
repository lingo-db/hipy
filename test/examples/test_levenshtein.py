import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
@hipy.compiled_function
def levenshtein_dist(a, b):
    """
    Compute the Levenshtein distance between two given
    strings (a and b). Taken from m.l. hetland
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n
    current = list(range(n+1))
    for i in range(1,m+1):
        previous, current = current, [i]+[0] * m
        for j in range(1, n+1):
            add, delete = previous[j] + 1, current[j-1] + 1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change +=1
            current[j] = min(add, delete, change)
    return current[n]

@hipy.compiled_function
def fn():
    print(levenshtein_dist("a", "b"))
    print(levenshtein_dist("kitten", not_constant("sitting")))
    print(levenshtein_dist(not_constant("kitten"), "sitting"))
    print(levenshtein_dist(not_constant("kitten"), not_constant("sitting")))

def test_levenshtein():
    check_prints(fn, """
1
3
3
3""",debug=False)
