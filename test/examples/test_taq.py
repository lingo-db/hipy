import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.statistics
from statistics import mean, stdev


@hipy.compiled_function
def find_peaks(y):
    lag = 100
    threshold = 10.0
    influence = 0.5

    t = len(y)
    signals = [0 for _ in range(t)]

    if t <= lag:
        return signals
    filtered_y = [y[i] if i < lag else 0. for i in range(t)]
    avg_filter = [0. for _ in range(t)]
    std_filter = [0. for _ in range(t)]
    avg_filter[lag] = mean(y[:lag])
    std_filter[lag] = stdev(y[:lag])

    for i in range(lag, t):
        if abs(y[i] - avg_filter[i - 1]) > threshold * std_filter[i - 1]:
            signals[i] = 1 if y[i] > avg_filter[i - 1] else -1
            filtered_y[i] = influence * y[i] + (1 - influence) * filtered_y[i - 1]
        else:
            signals[i] = 0
            filtered_y[i] = y[i]

        avg_filter[i] = mean(filtered_y[i - lag:i])
        std_filter[i] = stdev(filtered_y[i - lag:i])

    return signals


@hipy.compiled_function
def process_data(series):
    grouped = {}
    for bucket, volume in series:
        grouped[bucket] = grouped.get(bucket, 0) + volume

    y = [float(t[1]) for t in sorted(grouped.items())]
    return y, find_peaks(y)


@hipy.compiled_function
def taq(lines):
    BUCKET_SIZE = 1_000_000_000

    data = {}
    header = True

    for line in lines:
        if header:
            header = False
            continue
        x = line.split('|')
        if x[0] == 'END' or x[4] == 'ENDP':
            continue
        timestamp = int(x[0])
        symbol = x[2]
        volume = int(x[4])

        series = data.setdefault(symbol, [])
        series.append((timestamp // BUCKET_SIZE, volume))
    res = ""
    for symbol, series in data.items():
        y, signals = process_data(series)
        res = res + symbol + "," + str(sum(signals))
    return res


@hipy.compiled_function
def fn_taq():
    lines = [
        'Timestamp|Symbol|Message Category|Message Type|Message Sequence Number|Participant Identifier|Participant Timestamp|Financial Status|Currency Indicator|Security Status|Halt Reason|Related Security Indicator|Common Indicator|Last Price|Status Indicator|High Indication Price/Upper Limit Price Band|Low Indication Price/LowerL imit Price Band|Buy Volume|Sell Volume|Short Sale Restriction Indicator|Limit Up/Limit Down Indicator|Market Wide Circuit Breaker Decline Level 1|Market Wide Circuit Breaker Decline Level 2|Market Wide Circuit Breaker Decline Level 3|Market Wide Circuit Breaker Status|Message Text',
        '070005443462144||M|K|3036|S|070005443454208|||||||||||||||3334.62|3119.48|2868.49||',
        '070005443462400||M|K|6345|S|070005443454208|||||||||||||||3334.62|3119.48|2868.49||',
        '070005443462400||M|K|6571|S|070005443454208|||||||||||||||3334.62|3119.48|2868.49||',
        '070005443464448||M|K|6341|S|070005443451904|||||||||||||||3334.62|3119.48|2868.49||',
        '070005443464448||M|K|6935|S|070005443451904|||||||||||||||3334.62|3119.48|2868.49||',
        '070005443464704||M|K|8577|S|070005443454208|||||||||||||||3334.62|3119.48|2868.49||',
        '070005443464704||M|K|9257|S|070005443451904|||||||||||||||3334.62|3119.48|2868.49||']
    print(taq(lines))


def test_taq():
    check_prints(fn_taq, "M,0")
