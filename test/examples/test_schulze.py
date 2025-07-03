import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant


@hipy.compiled_function
def schulze(lines):
    # Strip superfluous information
    lines = [line[14:].rstrip() for line in lines]
    # lines =[line[14:] for line in lines]

    votes = [line.split(' ') for line in lines]

    # Make a canonical set of all the candidates
    candidates = {}
    for line in votes:
        for memberId in line:
            candidates[memberId] = 1

    # Go from member number to an index
    for i,k in enumerate(candidates):
        candidates[k] = i

    # And vice versa
    reverseCandidates = {}
    for k, v in candidates.items():
        reverseCandidates[v] = k

    # Turn the votes in to an index number
    numbers = [[candidates[memberId] for memberId in line] for line in votes]

    size = len(candidates)

    # Initialize the d and p matrixes
    row = []
    for i in range(size):
        row.append(0)

    d = []
    p = []
    for i in range(size):
        d.append(row[:])
        p.append(row[:])

    # Fill in the preferences in the d matrix
    for i in range(size):
        for line in numbers:

            for entry in line:
                if entry == i:
                    break
                d[entry][i] += 1

    # Calculate the p matrix. Algorithm copied straight from wikipedia
    # article http://en.wikipedia.org/wiki/Schulze_method
    for i in range(size):
        for j in range(size):
            if i != j:
                if d[i][j] > d[j][i]:
                    p[i][j] = d[i][j]
                else:
                    p[i][j] = 0

    for i in range(size):
        for j in range(size):
            if i != j:
                for k in range(size):
                    if i != k:
                        if j != k:
                            p[j][k] = max(p[j][k], min(p[j][i], p[i][k]))

    # Find the best candidate (p[candidate, X] >= p[X, candidate])
    # Put the candidate on the final list, remove the candidate from p and
    # repeat
    order = []
    still_candidate = [True for i in range(size)]
    while len(order) < size:
        for i in range(size):
            if still_candidate[i]:
                is_best = True
                for j in range(size):
                    if still_candidate[j]:
                        if i != j:
                            if p[j][i] > p[i][j]:
                                is_best = False
                if is_best:
                    order.append(i)
                    still_candidate[i] = False
    return order, reverseCandidates


@hipy.compiled_function
def fn_schulze():
    lines = not_constant(['227AB8KF2FVE  22270 72 3655 273 1 13425 3365 922 15866 18060 7073 19049 19821 ',
             '22DPWE323HPZ  63640 1 65252 62473 2607 56668 63081 13454 8715 378 13425 15353 4166 10796 11443 72 ',
             '236669ZH3P9A  13454 11443 16759 922 65252 3655 3419 10676 5204 13425 43932 378 1 42694 19049 64064 290 4166 ',
             '23B29NMXNQAV  56063 26576 11265 12346 3093 56668 18060 437 11072 12941 51716 63695 33137 15915 12468 63640 1135 34002 47292 61923 378 7073 ',
             '247AZ6WC6FKY  1 11443 378 11729 3419 7059 12393 437 1007 3 2608 967 ',
             '252HTEFP9YJ7  57909 44 16918 17337 25047 12393 17152 13776 12191 54098 23893 922 15866 16916 8790 63695 42694 34981 15074 53472 9876 9812 ',
             '254CTXDVX685  6315 1 58572 273 11443 16403 290 9360 1007 4699 7059 437 13425 22270 ',
             '26H3RZ9QPAXH  56475 19374 903 1343 13415 3419 12184 11265 20623 8715 17152 7073 1 39553 64304 43932 180 19049 63695 1306 54098 25047 ',
             '283QUQQR8HEC  1 11443 61657 15175 1007 31395 44 ',
             '287XCAH8R458  1 11443 1981 3 3365 17591 4166 15915 967 3655 3419 ',
             '28MXV6UBHERV  1 11443 378 15915 4166 3 13425 3655 1007 3419 65252 7059 1981 10676 9951 8715 11729 17591 437 22270 61657 72 ',
             '28RB4DTEY436  17743 1 11443 ', '293882MU6J3B  10796 '])
    order, reverseCandidates = schulze(lines)
    print(",".join([reverseCandidates[i] for i in order]))

def test_schulze():
        check_prints(fn_schulze,"""1,11443,378,13425,4166,15915,3,3655,3419,1007,437,63695,8715,7059,65252,72,7073,922,22270,19049,43932,11265,17591,11729,54098,17152,25047,44,10676,56668,61657,13454,3365,273,12393,1981,63640,15866,18060,42694,967,10796,16759,5204,290,56063,26576,2608,56475,15175,19374,17743,6315,9951,31395,64064,57909,62473,2607,63081,19821,12346,3093,11072,903,58572,16918,15353,12941,1343,51716,16403,17337,33137,13415,13776,12468,12184,9360,12191,20623,4699,1135,39553,34002,47292,61923,64304,23893,16916,8790,180,34981,1306,15074,53472,9876,9812""")
