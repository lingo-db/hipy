"""Tests for UDFs from the qure-aggr-hipy benchmark
(`udf-benchmark/benchmarks/lingodb/qure-aggr-hipy/initialize.sql`).

This benchmark's UDF set is a superset of qure-hipy's. The shared UDFs
(q1..q5, q7..q9, q12..q16, q18..q27, q2) are identical between the two
files and are already tested in `test_qure_udfs.py`. This file covers
the four extra UDFs that appear only here: `q6_udf`, `q10_udf`,
`q11_udf`, `q17_udf`.
"""
import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.datetime
import datetime


@hipy.compiled_function
def q10_udf(key):
    mk = str.maketrans("ai", "zz")
    s1 = key.lower()
    s2 = s1.translate(mk)
    return s2


@hipy.compiled_function
def fn_q10():
    print(q10_udf(not_constant("Apple Pie")))
    print(q10_udf(not_constant("BANANA")))
    print(q10_udf(not_constant("Train")))


def test_q10():
    # SQL declares q10 as `LANGUAGE hipy_fallback` — `str.maketrans` and
    # `str.translate` aren't natively compiled, so we fall back to CPython.
    check_prints(fn_q10, """
zpple pze
bznznz
trzzn
""", fallback=True)


@hipy.compiled_function
def q11_udf(l_shipinstruct):

    instruction_cleaned = l_shipinstruct.strip().lower()

    urgency_keywords = ['urgent', 'priority', 'express']
    standard_keywords = ['standard', 'normal', 'routine']

    urgent_count = sum(instruction_cleaned.count(word) for word in urgency_keywords)
    standard_count = sum(instruction_cleaned.count(word) for word in standard_keywords)

    if urgent_count > 0:
        urgency_level = "High Urgency"
    elif standard_count > 0:
        urgency_level = "Standard"
    else:
        urgency_level = "Unknown"

    words = instruction_cleaned.split(" ")
    word_count = len(words)
    if word_count > 10:
        complexity = "Complex Instruction"
    else:
        complexity = "Simple Instruction"

    instruction_summary = "Instruction: " + l_shipinstruct + " " + "Urgency Level: " + urgency_level+ " Word Count: " + str(word_count) + ", Complexity: " + complexity + " Urgent Keyword Count: " + str(urgent_count) +" , Standard Keyword Count: " + str(standard_count)

    is_fragile = 'fragile' in instruction_cleaned
    if is_fragile :
        instruction_summary = instruction_summary + " Warning: The instruction mentions [fragile] - Handle with care."

    no_bending = 'do not bend' in instruction_cleaned
    if no_bending:
        instruction_summary = instruction_summary + " Important: Instruction includes [Do not bend]."

    if urgent_count == 0 and standard_count == 0:
        instruction_summary = instruction_summary  + " No urgency keywords found in the instruction."

    return instruction_summary


@hipy.compiled_function
def fn_q11():
    print(q11_udf(not_constant("urgent handle with care do not bend fragile package please")))
    print(q11_udf(not_constant("standard delivery")))
    print(q11_udf(not_constant("just a regular comment")))


def test_q11():
    check_prints(fn_q11, """
Instruction: urgent handle with care do not bend fragile package please Urgency Level: High Urgency Word Count: 10, Complexity: Simple Instruction Urgent Keyword Count: 1 , Standard Keyword Count: 0 Warning: The instruction mentions [fragile] - Handle with care. Important: Instruction includes [Do not bend].
Instruction: standard delivery Urgency Level: Standard Word Count: 2, Complexity: Simple Instruction Urgent Keyword Count: 0 , Standard Keyword Count: 1
Instruction: just a regular comment Urgency Level: Unknown Word Count: 4, Complexity: Simple Instruction Urgent Keyword Count: 0 , Standard Keyword Count: 0 No urgency keywords found in the instruction.
""")


@hipy.compiled_function
def q17_udf(l_shipmode, l_shipdate, l_receiptdate, l_commitdate, l_returnflag):
    if not l_shipmode or not l_shipdate or not l_receiptdate or not l_commitdate:
        return "Missing necessary data"

    days_between_ship_and_receipt = (l_receiptdate - l_shipdate).days
    days_to_commit = (l_commitdate - l_shipdate).days

    if days_between_ship_and_receipt <= days_to_commit:
        performance = "On-time"
    else:
        performance = "Delayed"

    if l_returnflag == 'R':
        return_status = "Returned"
    else:
        return_status = "Not Returned"

    shipping_class = ''
    if 'AIR' in l_shipmode.upper():
        shipping_class = 'Air Shipment'
    elif 'RAIL' in l_shipmode.upper():
        shipping_class = 'Rail Shipment'
    else:
        shipping_class = 'Other Shipment'

    final_report = (
        f"Shipping Mode: {shipping_class}\n"
        f"Days Between Ship and Receipt: {days_between_ship_and_receipt}\n"
        f"Days to Commit: {days_to_commit}\n"
        f"Performance: {performance}\n"
        f"Return Status: {return_status}"
    )

    if l_returnflag == 'R':
        return_penalty = days_between_ship_and_receipt * 0.1
        final_report += f"\nReturn Penalty (in days): {return_penalty:.2f}"

    if days_between_ship_and_receipt > 30:
        final_report += "\nWarning: Excessive delay in shipment."

    return final_report


@hipy.compiled_function
def fn_q17():
    ship = datetime.date(not_constant(2024), not_constant(1), not_constant(1))
    receipt_fast = datetime.date(not_constant(2024), not_constant(1), not_constant(4))
    receipt_delayed = datetime.date(not_constant(2024), not_constant(1), not_constant(15))
    receipt_slow = datetime.date(not_constant(2024), not_constant(3), not_constant(1))
    commit = datetime.date(not_constant(2024), not_constant(1), not_constant(10))
    print(q17_udf(not_constant("AIR"), ship, receipt_fast, commit, not_constant("R")))
    print("---")
    print(q17_udf(not_constant("RAIL"), ship, receipt_delayed, commit, not_constant("N")))
    print("---")
    print(q17_udf(not_constant("TRUCK"), ship, receipt_slow, commit, not_constant("R")))


def test_q17():
    check_prints(fn_q17, """
Shipping Mode: Air Shipment
Days Between Ship and Receipt: 3
Days to Commit: 9
Performance: On-time
Return Status: Returned
Return Penalty (in days): 0.30
---
Shipping Mode: Rail Shipment
Days Between Ship and Receipt: 14
Days to Commit: 9
Performance: Delayed
Return Status: Not Returned
---
Shipping Mode: Other Shipment
Days Between Ship and Receipt: 60
Days to Commit: 9
Performance: Delayed
Return Status: Returned
Return Penalty (in days): 6.00
Warning: Excessive delay in shipment.
""")


@hipy.compiled_function
def q6_udf(l_comment):
    if not l_comment:
        return 'No comment provided'
    sensitive_words = ['confidential', 'proprietary', 'secret']
    scrambled_comment = l_comment
    for word in sensitive_words:
        scrambled_comment = scrambled_comment.replace(word, '*' * len(word))
    words = scrambled_comment.split()
    scrambled_words = [''.join(sorted(word)) for word in words]
    scrambled_final = ' '.join(scrambled_words)
    return f"Masked and Scrambled: {scrambled_final}"


@hipy.compiled_function
def fn_q6():
    print(q6_udf(not_constant("")))
    print(q6_udf(not_constant("hello confidential world")))
    print(q6_udf(not_constant("normal text here")))


def test_q6():
    check_prints(fn_q6, """
No comment provided
Masked and Scrambled: ehllo ************ dlorw
Masked and Scrambled: almnor ettx eehr
""")
