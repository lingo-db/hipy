"""Tests for UDFs from the qure-hipy benchmark
(`udf-benchmark/benchmarks/lingodb/qure-hipy/initialize.sql`).
"""
import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.math
import math
from math import sqrt, acos, atan2, cos, sin, exp
import hipy.lib.datetime
import datetime


@hipy.compiled_function
def q12_udf(key):
    key_list = key.split(" ")
    code = ["A", "I"]
    filtered_list = []
    for c in key_list:
        if c in code:
            filtered_list.append(c)
    return "".join(filtered_list)


@hipy.compiled_function
def fn_q12():
    print(q12_udf(not_constant("A B I C A")))
    print("[" + q12_udf(not_constant("B C D")) + "]")
    print(q12_udf(not_constant("A I A I I")))


def test_q12():
    check_prints(fn_q12, """
AIA
[]
AIAII
""")


@hipy.compiled_function
def q13_udf(l_comment):
    comment_length = len(l_comment)
    word_count = len(l_comment.split(" "))
    cleaned_comment_temp = l_comment.lower().replace('urgent', '')
    cleaned_comment = cleaned_comment_temp.replace('important', '').strip()
    vowel_count = sum(1 for char in cleaned_comment if char in ['a','e','i','o','u'])
    is_damaged = 'damaged' in cleaned_comment
    is_late = 'late' in cleaned_comment
    if is_damaged:
        status = 'Item Damaged'
    elif is_late:
        status = 'Late Delivery'
    else:
        status = 'No Issues'
    analysis_summary = "Original Comment: " + l_comment + " " + "Cleaned Comment: " + cleaned_comment + " " + "Length: " + str(comment_length) + ", Words: " + str(word_count) +", Vowels: " + str(vowel_count) + " " + "Status: " + status
    return analysis_summary


@hipy.compiled_function
def fn_q13():
    print(q13_udf(not_constant("Urgent damaged items")))
    print(q13_udf(not_constant("Nothing special")))
    print(q13_udf(not_constant("late important shipment")))


def test_q13():
    check_prints(fn_q13, """
Original Comment: Urgent damaged items Cleaned Comment: damaged items Length: 20, Words: 3, Vowels: 5 Status: Item Damaged
Original Comment: Nothing special Cleaned Comment: nothing special Length: 15, Words: 2, Vowels: 5 Status: No Issues
Original Comment: late important shipment Cleaned Comment: late  shipment Length: 23, Words: 3, Vowels: 4 Status: Late Delivery
""")


@hipy.compiled_function
def q14_udf(l_quantity, l_extendedprice):
    a = 1.0
    b = 1.0
    term1 = (l_quantity / a) ** 2.0 + (l_extendedprice / b) ** 2.0
    if term1 >= 1.0:
        root_term = sqrt(term1 - 1.0)
    else:
        root_term = 0.0
    eccentricity = sqrt(1.0 - (b / a) ** 2.0)
    if eccentricity == 0.0:
        eccentricity = 1e-6
    distance = a * acos(min(1.0, root_term / eccentricity))
    angle = atan2(l_extendedprice, l_quantity)
    projection_x = a * cos(angle)
    projection_y = b * sin(angle)
    final_distance = sqrt((l_quantity - projection_x) ** 2.0 + (l_extendedprice - projection_y) ** 2.0)
    return round(final_distance, 4)


@hipy.compiled_function
def fn_q14():
    print(q14_udf(not_constant(1.0), not_constant(0.0)))
    print(q14_udf(not_constant(3.0), not_constant(4.0)))
    print(q14_udf(not_constant(0.0), not_constant(0.0)))


def test_q14():
    check_prints(fn_q14, """
0.0
4.0
1.0
""")


@hipy.compiled_function
def q15_udf(l_comment):
    sensitive_words = ['damaged', 'late', 'urgent', 'confidential']
    positive_keywords = ['excellent', 'good', 'satisfied']
    negative_keywords = ['bad', 'delayed', 'broken', 'terrible']
    comment_cleaned = l_comment.strip().lower()
    positive_count = 0
    negative_count = 0
    for word in sensitive_words:
        comment_cleaned = comment_cleaned.replace(word, '******')
    for word_pos in positive_keywords:
        positive_count += comment_cleaned.count(word_pos)
    for word_neg in negative_keywords:
        negative_count += comment_cleaned.count(word_neg)
    if positive_count > negative_count:
        sentiment = "Positive"
    elif negative_count > positive_count:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    redacted_comment = ''
    l_comment_split = l_comment.split(" ")
    for word_redacted in l_comment_split:
        word_redacted_lower = word_redacted.lower()
        is_sensitive = word_redacted.lower() in sensitive_words
        if is_sensitive:
            redacted_comment = redacted_comment + '****** '
        else:
            redacted_comment = redacted_comment + word_redacted
            redacted_comment = redacted_comment + ' '
    summary = "Original Comment: " + l_comment + " " + "Redacted Comment: " + redacted_comment.strip() + " " + "Sentiment: " + sentiment + " " + "Positive Words: " + str(positive_count) + ", Negative Words: " + str(negative_count)
    is_damaged =  'damaged' in l_comment.lower()
    is_late = 'late' in l_comment.lower()
    if is_damaged:
        summary = summary + " Warning: The comment mentions a damaged item."
    if is_late:
        summary = summary + " Warning: The comment mentions late delivery."
    is_confidential = 'confidential' in l_comment.lower()
    if is_confidential:
        summary = summary + "Confidential content detected: Handle with care."
    return summary


@hipy.compiled_function
def fn_q15():
    print(q15_udf(not_constant("good product but late")))
    print(q15_udf(not_constant("excellent service")))
    print(q15_udf(not_constant("damaged and terrible")))
    print(q15_udf(not_constant("nothing special")))


def test_q15():
    check_prints(fn_q15, """
Original Comment: good product but late Redacted Comment: good product but ****** Sentiment: Positive Positive Words: 1, Negative Words: 0 Warning: The comment mentions late delivery.
Original Comment: excellent service Redacted Comment: excellent service Sentiment: Positive Positive Words: 1, Negative Words: 0
Original Comment: damaged and terrible Redacted Comment: ****** and terrible Sentiment: Negative Positive Words: 0, Negative Words: 1 Warning: The comment mentions a damaged item.
Original Comment: nothing special Redacted Comment: nothing special Sentiment: Neutral Positive Words: 0, Negative Words: 0
""")


@hipy.compiled_function
def q16_udf(l_shipmode, l_returnflag):
    shipmode_status = l_shipmode.upper()
    if l_returnflag == 'R':
        return_status = 'Returned'
    else:
        return_status = 'Not Returned'
    in_air = 'AIR' in shipmode_status
    on_rail = 'RAIL' in shipmode_status
    if in_air:
        transport_mode = "Air Transport"
    else:
        if on_rail:
            transport_mode = "Rail Transport"
        else:
            transport_mode = "Other Transport"
    result_summary = "Shipping Mode: " + transport_mode + ", Return Status: " + return_status
    return result_summary


@hipy.compiled_function
def fn_q16():
    print(q16_udf(not_constant("AIR"), not_constant("R")))
    print(q16_udf(not_constant("rail"), not_constant("N")))
    print(q16_udf(not_constant("TRUCK"), not_constant("R")))


def test_q16():
    check_prints(fn_q16, """
Shipping Mode: Air Transport, Return Status: Returned
Shipping Mode: Rail Transport, Return Status: Not Returned
Shipping Mode: Other Transport, Return Status: Returned
""")


@hipy.compiled_function
def q18_udf(o_totalprice):
    period = 0.01
    term_1 = (1.0 / 1.0) * math.sin(2.0 * math.pi * 1.0 * o_totalprice / period)
    term_2 = (1.0 / 2.0) * math.sin(2.0 * math.pi * 2.0 * o_totalprice / period)
    term_3 = (1.0 / 3.0) * math.sin(2.0 * math.pi * 3.0 * o_totalprice / period)
    term_4 = (1.0 / 4.0) * math.sin(2.0 * math.pi * 4.0 * o_totalprice / period)
    term_5 = (1.0 / 5.0) * math.sin(2.0 * math.pi * 5.0 * o_totalprice / period)
    result = term_1 + term_2 + term_3 + term_4 + term_5
    price_adjustment = math.cos(o_totalprice / period) * 0.05
    result += price_adjustment
    normalization_factor = 1.0 + abs(o_totalprice) * 0.01
    final_result = result / normalization_factor
    return round(final_result, 4)


@hipy.compiled_function
def fn_q18():
    print(q18_udf(not_constant(0.0)))


def test_q18():
    check_prints(fn_q18, """
0.05
""")


@hipy.compiled_function
def q19_udf(l_orderkey, l_partkey, l_shipmode):
    if not l_shipmode:
        return 'Missing ship mode'
    if 'VIP' in l_shipmode.upper():
        mode_abbrev = l_shipmode[:3].upper() + '*'
    else:
        mode_abbrev = ''
    id_part1 = str(l_orderkey).zfill(6)
    id_part2 = str(l_partkey).zfill(6)
    unique_id = f"{id_part1}-{id_part2}-{mode_abbrev}"
    if len(l_shipmode) > 5:
        extra_code = l_shipmode[-2:].upper()
        unique_id += f"-{extra_code}"
    return unique_id


@hipy.compiled_function
def fn_q19():
    print(q19_udf(not_constant(1), not_constant(2), not_constant("")))
    print(q19_udf(not_constant(1), not_constant(2), not_constant("AIR")))
    print(q19_udf(not_constant(100), not_constant(200), not_constant("VIPFAST")))
    print(q19_udf(not_constant(100), not_constant(200), not_constant("TRUCK")))
    print(q19_udf(not_constant(100), not_constant(200), not_constant("REGULAR")))


def test_q19():
    # The SQL file marks q19 with `LANGUAGE hipy_fallback` — it uses
    # `str.zfill(...)`, which is routed through the Python fallback.
    check_prints(fn_q19, """
Missing ship mode
000001-000002-
000100-000200-VIP*-ST
000100-000200-
000100-000200--AR
""", fallback=True)


@hipy.compiled_function
def q1_udf(l_extendedprice, l_discount, l_quantity, l_returnflag, l_tax):
    if l_extendedprice < 0.0 or l_discount < 0.0 or l_discount > 1.0 or l_tax < 0.0:
        profit_report = "Invalid input data"
        return profit_report
    price_after_discount = l_extendedprice * (1.0 - l_discount)
    if l_quantity > 50.0:
        bulk_discount = 0.05
        price_after_discount = price_after_discount * (1.0 - bulk_discount)
    else:
        bulk_discount = 0.0
    price_with_tax = price_after_discount * (1.0 + l_tax)
    profit_margin = 0.2
    profit = price_with_tax * profit_margin
    profit_report = (
        "Original Price: $" + str(l_extendedprice) +
        " Discount Applied: " + str(l_discount * 100.0) + "%" +
        " Bulk Discount Applied: " + str(bulk_discount * 100.0) + "%" +
        " Price After Discounts: $" + str(price_after_discount) +
        " Tax Rate: " + str(l_tax * 100.0) + "%" +
        " Price After Tax: $" + str(price_with_tax) +
        " Profit: $" + str(profit) +
        " (Profit Margin: " + str(profit_margin * 100.0) + "%)"
    )
    if l_returnflag == 'R':
        return_penalty = price_with_tax * 0.1
        profit_report += (
            " Return Penalty: $" + str(return_penalty) +
            " Profit after Penalty: $" + str(profit - return_penalty)
        )
    if l_quantity > 50.0:
        profit_report += " Bulk Order Detected: Extra Discounts Applied."
    return profit_report


@hipy.compiled_function
def fn_q1():
    # Inputs chosen to avoid binary-float rounding in str(float).
    print(q1_udf(not_constant(200.0), not_constant(0.5), not_constant(100.0), not_constant("R"), not_constant(0.0)))
    print(q1_udf(not_constant(-1.0), not_constant(0.1), not_constant(10.0), not_constant("R"), not_constant(0.05)))
    print(q1_udf(not_constant(100.0), not_constant(0.5), not_constant(10.0), not_constant("N"), not_constant(0.0)))


def test_q1():
    check_prints(fn_q1, """
Original Price: $200.0 Discount Applied: 50.0% Bulk Discount Applied: 5.0% Price After Discounts: $95.0 Tax Rate: 0.0% Price After Tax: $95.0 Profit: $19.0 (Profit Margin: 20.0%) Return Penalty: $9.5 Profit after Penalty: $9.5 Bulk Order Detected: Extra Discounts Applied.
Invalid input data
Original Price: $100.0 Discount Applied: 50.0% Bulk Discount Applied: 0.0% Price After Discounts: $50.0 Tax Rate: 0.0% Price After Tax: $50.0 Profit: $10.0 (Profit Margin: 20.0%)
""")


@hipy.compiled_function
def q20_udf(l_extendedprice, l_discount, l_quantity):
    if l_discount < 0.0 or l_discount > 1.0 or l_extendedprice < 0.0:
        result = 'Invalid price or discount'
        return result
    final_price = l_extendedprice * (1.0 - l_discount)
    if l_quantity > 50.0:
        bulk_status = 'Bulk Order'
    else:
        bulk_status ='Regular Order'
    if l_discount < 0.1:
        discount_category = 'Low Discount'
    else:
        discount_category = 'High Discount'
    result1 = 'Final Price: ' + str(final_price) + ', ' +  'Quantity: ' + str(l_quantity) + ', ' + 'Order Type: ' + bulk_status + ', ' +  'Discount Category: ' + discount_category
    return result1


@hipy.compiled_function
def fn_q20():
    print(q20_udf(not_constant(100.0), not_constant(0.5), not_constant(100.0)))
    print(q20_udf(not_constant(100.0), not_constant(0.05), not_constant(10.0)))
    print(q20_udf(not_constant(-1.0), not_constant(0.1), not_constant(10.0)))


def test_q20():
    check_prints(fn_q20, """
Final Price: 50.0, Quantity: 100.0, Order Type: Bulk Order, Discount Category: High Discount
Final Price: 95.0, Quantity: 10.0, Order Type: Regular Order, Discount Category: Low Discount
Invalid price or discount
""")


@hipy.compiled_function
def q21_udf(o_totalprice, l_quantity):
    growth_rate = 0.03
    carrying_capacity = 100
    time = 5
    decay_rate = 0.02
    external_factor = 1.5
    logistic_growth = (carrying_capacity * l_quantity) / (
        l_quantity + (carrying_capacity - l_quantity) * math.exp(-growth_rate * time)
    )
    decay_factor = math.exp(-decay_rate * time)
    total_growth = logistic_growth * decay_factor
    external_adjustment = external_factor * o_totalprice * 0.05
    adjusted_growth = total_growth + external_adjustment - (total_growth ** 2.0) * 0.01
    return round(adjusted_growth, 4)


@hipy.compiled_function
def fn_q21():
    print(q21_udf(not_constant(100.0), not_constant(10.0)))
    print(q21_udf(not_constant(0.0), not_constant(50.0)))


def test_q21():
    check_prints(fn_q21, """
16.775
24.9812
""")


@hipy.compiled_function
def q22_udf(l_shipmode, l_returnflag):
    shipmode_status = l_shipmode.upper()
    if l_returnflag == 'R':
        return_status = 'Returned'
    else:
        return_status = 'Not Returned'
    is_air = 'AIR' in shipmode_status
    is_rail = 'RAIL' in shipmode_status
    if is_air:
        transport_mode = "Air Transport"
    elif is_rail:
        transport_mode = "Rail Transport"
    else:
        transport_mode = "Other Transport"
    result_summary = "Shipping Mode: " + transport_mode + ", Return Status: " + return_status
    return result_summary


@hipy.compiled_function
def fn_q22():
    print(q22_udf(not_constant("AIR"), not_constant("R")))
    print(q22_udf(not_constant("rail"), not_constant("N")))
    print(q22_udf(not_constant("TRUCK"), not_constant("R")))


def test_q22():
    check_prints(fn_q22, """
Shipping Mode: Air Transport, Return Status: Returned
Shipping Mode: Rail Transport, Return Status: Not Returned
Shipping Mode: Other Transport, Return Status: Returned
""")


@hipy.compiled_function
def q23_udf(l_comment):
    cleaned_comment = l_comment.lower().replace('!', '').replace('@', '')
    if 'special' in cleaned_comment  or 'urgent' in cleaned_comment:
        summary = "Keyword found in comment: " +cleaned_comment
    else:
        summary = "No special keywords in comment: "+cleaned_comment
    return summary


@hipy.compiled_function
def fn_q23():
    print(q23_udf(not_constant("Special!! offer")))
    print(q23_udf(not_constant("urgent @item")))
    print(q23_udf(not_constant("normal comment")))


def test_q23():
    check_prints(fn_q23, """
Keyword found in comment: special offer
Keyword found in comment: urgent item
No special keywords in comment: normal comment
""")


@hipy.compiled_function
def q24_udf(l_shipmode, l_returnflag):
    shipmode_upper = l_shipmode.strip().upper()
    if l_returnflag == 'R':
        return_status = 'Returned'
    else:
        return_status = 'Not Returned'
    result_summary = "Mode: " + shipmode_upper + ", Return Status: " + return_status
    return result_summary


@hipy.compiled_function
def fn_q24():
    print(q24_udf(not_constant("  air  "), not_constant("R")))
    print(q24_udf(not_constant("truck"), not_constant("N")))


def test_q24():
    check_prints(fn_q24, """
Mode: AIR, Return Status: Returned
Mode: TRUCK, Return Status: Not Returned
""")


@hipy.compiled_function
def q25_udf(l_shipmode):
    formatted_mode = l_shipmode.strip().upper()
    is_urgent = True
    if is_urgent:
        urgency_status = 'URGENT'
    else:
        urgency_status = 'Non-urgent'
    in_air = 'AIR' in formatted_mode
    on_rail = 'RAIL' in formatted_mode
    if in_air:
        transport = 'Air transport ( ' + urgency_status + ')'
    else:
        if on_rail:
            transport = 'Rail transport ( ' + urgency_status + ')'
        else:
            transport = 'Other transport ( ' + urgency_status + ')'
    return transport


@hipy.compiled_function
def fn_q25():
    print(q25_udf(not_constant(" air ")))
    print(q25_udf(not_constant("rail")))
    print(q25_udf(not_constant("truck")))


def test_q25():
    check_prints(fn_q25, """
Air transport ( URGENT)
Rail transport ( URGENT)
Other transport ( URGENT)
""")


@hipy.compiled_function
def q26_udf(l_shipinstruct, l_comment):
    cleaned_instruction = l_shipinstruct.lower().strip()
    cleaned_comment = l_comment.upper().strip()
    instruction_length = len(cleaned_instruction)
    comment_length = len(cleaned_comment)
    if instruction_length > 20:
        instruction_summary = f"Instruction: {cleaned_instruction[:20]}... (truncated)"
    else:
        instruction_summary = f"Instruction: {cleaned_instruction}"
    if comment_length > 30:
        comment_summary = f"Comment: {cleaned_comment[:30]}... (truncated)"
    else:
        comment_summary = f"Comment: {cleaned_comment}"
    result = f"{instruction_summary}\n{comment_summary}"
    return result


@hipy.compiled_function
def fn_q26():
    print(q26_udf(not_constant("short instruction"), not_constant("short comment")))
    print("---")
    print(q26_udf(not_constant("A very long shipping instruction that exceeds twenty characters"),
                  not_constant("A long comment with more than thirty characters")))


def test_q26():
    check_prints(fn_q26, """
Instruction: short instruction
Comment: SHORT COMMENT
---
Instruction: a very long shipping... (truncated)
Comment: A LONG COMMENT WITH MORE THAN ... (truncated)
""")


@hipy.compiled_function
def q27_udf(l_shipmode):
    key_list = l_shipmode.split(" ")
    if key_list[0] == "REG":
        return "regular"
    if key_list[0] in ["AIR", "MAIL"]:
        return "fast"
    return "slow"


@hipy.compiled_function
def fn_q27():
    print(q27_udf(not_constant("REG AIR")))
    print(q27_udf(not_constant("AIR")))
    print(q27_udf(not_constant("MAIL")))
    print(q27_udf(not_constant("TRUCK")))


def test_q27():
    check_prints(fn_q27, """
regular
fast
fast
slow
""")


@hipy.compiled_function
def q2_udf(o_totalprice, l_extendedprice):
    decay_rate1 = 0.02
    decay_rate2 = 0.5
    factor1 = 0.5
    factor2 = 0.5
    external_factor = 1.5
    term1 = o_totalprice * exp(-decay_rate1) * factor1
    term2 = l_extendedprice * exp(-decay_rate2) * factor2
    total_decay = term1 + term2
    decay_adjustment = sqrt(term1**2.0 + term2**2.0)
    adjusted_value = total_decay * decay_adjustment
    if adjusted_value > 0.0:
        result_value = round(adjusted_value, 4)
    else:
        result_value = 0.0
    return result_value


@hipy.compiled_function
def fn_q2():
    print(q2_udf(not_constant(10.0), not_constant(5.0)))
    print(q2_udf(not_constant(-5.0), not_constant(-5.0)))


def test_q2():
    check_prints(fn_q2, """
32.9222
0.0
""")


@hipy.compiled_function
def q3_udf(l_shipinstruct):
    instruction_lower = l_shipinstruct.lower().strip()
    f_express = 'express' in instruction_lower
    f_standard = 'standard' in instruction_lower
    if f_express:
        shipping_speed = 'Express Delivery'
    else:
        if f_standard:
            shipping_speed = 'Standard Delivery'
        else:
            shipping_speed = 'Unknown Delivery'
    keyword_count = instruction_lower.count('deliver')
    result1 = "Shipping Speed: " + shipping_speed + ", Keyword deliver Count: " + str(keyword_count)
    return result1


@hipy.compiled_function
def fn_q3():
    print(q3_udf(not_constant("Please deliver express")))
    print(q3_udf(not_constant("standard delivery")))
    print(q3_udf(not_constant("unknown route")))


def test_q3():
    check_prints(fn_q3, """
Shipping Speed: Express Delivery, Keyword deliver Count: 1
Shipping Speed: Standard Delivery, Keyword deliver Count: 1
Shipping Speed: Unknown Delivery, Keyword deliver Count: 0
""")


@hipy.compiled_function
def q4_udf(key):
    shipmode_list = [
        "AIR",
        "MAIL",
        "RAIL",
        "SHIP",
        "TRUCK",
        "REG AIR",
    ]
    if key in shipmode_list:
        return shipmode_list.index(key)
    else:
        return -1


@hipy.compiled_function
def fn_q4():
    print(q4_udf(not_constant("AIR")))
    print(q4_udf(not_constant("RAIL")))
    print(q4_udf(not_constant("REG AIR")))
    print(q4_udf(not_constant("OTHER")))


def test_q4():
    check_prints(fn_q4, """
0
2
5
-1
""")


@hipy.compiled_function
def q5_udf(l_shipinstruct):
    cleaned_instruction = l_shipinstruct.strip().upper()
    is_urgent = 'URGENT' in cleaned_instruction
    is_standard = 'STANDARD' in cleaned_instruction
    if is_urgent:
        urgency = 'High Urgency'
    elif is_standard:
        urgency = 'Standard'
    else:
        urgency = 'Unknown Urgency'
    word_count = len(cleaned_instruction.split(" "))
    if word_count > 5:
        summary = "Complex instruction"
    else:
        summary = "Simple instruction"
    result_summary = "Cleaned Instruction: " + cleaned_instruction + ", Urgency: " + urgency + ", Words: " + str(word_count) +", Summary: " + summary
    return result_summary


@hipy.compiled_function
def fn_q5():
    print(q5_udf(not_constant("urgent handling")))
    print(q5_udf(not_constant("standard delivery please")))
    print(q5_udf(not_constant("this is a very long complex instruction with many words")))


def test_q5():
    check_prints(fn_q5, """
Cleaned Instruction: URGENT HANDLING, Urgency: High Urgency, Words: 2, Summary: Simple instruction
Cleaned Instruction: STANDARD DELIVERY PLEASE, Urgency: Standard, Words: 3, Summary: Simple instruction
Cleaned Instruction: THIS IS A VERY LONG COMPLEX INSTRUCTION WITH MANY WORDS, Urgency: Unknown Urgency, Words: 10, Summary: Complex instruction
""")


@hipy.compiled_function
def q7_udf(l_extendedprice, l_discount, l_quantity):
    final_price = l_extendedprice * (1.0 - l_discount)
    if l_quantity > 50.0:
        final_price = final_price * 0.9
    if l_discount < 0.1:
        discount_category = 'Low'
    else:
        discount_category = 'High'
    result_summary = "Original Price: + " + str(l_extendedprice) +  "Discount: " + str(l_discount) + " (" + discount_category + "), " + "Quantity: " + str(l_quantity) + ", " + "Final Price: " + str(final_price)
    return result_summary


@hipy.compiled_function
def fn_q7():
    print(q7_udf(not_constant(100.0), not_constant(0.5), not_constant(100.0)))
    print(q7_udf(not_constant(100.0), not_constant(0.05), not_constant(10.0)))


def test_q7():
    check_prints(fn_q7, """
Original Price: + 100.0Discount: 0.5 (High), Quantity: 100.0, Final Price: 45.0
Original Price: + 100.0Discount: 0.05 (Low), Quantity: 10.0, Final Price: 95.0
""")


@hipy.compiled_function
def q8_udf(l_shipdate, l_receiptdate):
    days_between = (l_receiptdate - l_shipdate).days
    if days_between < 0:
        return 'Error in dates'
    if days_between <= 5:
        classification = 'Quick Shipment'
    elif days_between <= 10:
        classification = 'Normal Shipment'
    else:
        classification = 'Slow Shipment'
    result_summary = "Days Between: " + str(days_between) + ", Classification: " + classification
    return result_summary


@hipy.compiled_function
def fn_q8():
    d1 = datetime.date(not_constant(2024), not_constant(1), not_constant(1))
    d2 = datetime.date(not_constant(2024), not_constant(1), not_constant(3))
    d3 = datetime.date(not_constant(2024), not_constant(1), not_constant(8))
    d4 = datetime.date(not_constant(2024), not_constant(1), not_constant(20))
    print(q8_udf(d1, d2))
    print(q8_udf(d1, d3))
    print(q8_udf(d1, d4))
    print(q8_udf(d4, d1))


def test_q8():
    check_prints(fn_q8, """
Days Between: 2, Classification: Quick Shipment
Days Between: 7, Classification: Normal Shipment
Days Between: 19, Classification: Slow Shipment
Error in dates
""")


@hipy.compiled_function
def q9_udf(l_comment):
    l_comment = l_comment.strip()
    words = l_comment.split(" ")
    wordlist = (word[0].upper() for word in words if word != '')
    initials = ''.join(wordlist)
    word_count = len(words)
    if word_count > 20:
        complexity = "High complexity"
    elif word_count > 10:
        complexity = "Medium complexity"
    else:
        complexity = "Low complexity"
    return "Initials: " + initials + ", Word Count: " + str(word_count) +", Complexity: " + complexity


@hipy.compiled_function
def fn_q9():
    print(q9_udf(not_constant("hello world foo bar")))
    print(q9_udf(not_constant("a b c d e f g h i j k l")))
    print(q9_udf(not_constant("a b c d e f g h i j k l m n o p q r s t u v")))


def test_q9():
    check_prints(fn_q9, """
Initials: HWFB, Word Count: 4, Complexity: Low complexity
Initials: ABCDEFGHIJKL, Word Count: 12, Complexity: Medium complexity
Initials: ABCDEFGHIJKLMNOPQRSTUV, Word Count: 22, Complexity: High complexity
""")
