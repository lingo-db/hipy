import random
import re

import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.pandas
import pandas as pd

from hipy.value import raw_module

_pd = raw_module(pd)


@hipy.compiled_function
def gen_data():
    return pd.DataFrame.from_dict({
        "line": [
            """83.149.9.216 - - [17/May/2015:10:05:03 +0000] "GET /presentations/logstash-monitorama-2013/images/kibana-search.png HTTP/1.1" 200 203023 "http://semicomplete.com/presentations/logstash-monitorama-2013/" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1700.77 Safari/537.36" """]
    })


def RandomizeEndpointUDF(x) -> str:
    return re.sub('^/~[^/]+', '/~' + ''.join([random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for t in range(10)]), x)


@hipy.compiled_function
def logs(df):
    try_int = lambda x: intrinsics.try_or_default(lambda: int(x), 0)
    df["cols"] = df["line"].apply(lambda x: x.split(' '))
    df["ip"] = df['cols'].apply(lambda x: x[0].strip() if len(x) > 0 else '')
    df["client_id"] = df['cols'].apply(lambda x: x[1].strip() if len(x) > 1 else '')
    df["user_id"] = df['cols'].apply(lambda x: x[2].strip() if len(x) > 2 else '')
    df["endpoint"] = df['cols'].apply(lambda x: x[6].strip() if len(x) > 6 else '')
    df["date"] = df['cols'].apply(lambda x: x[3] + " " + x[4] if len(x) > 4 else '')
    df["date"] = df['date'].apply(lambda x: x.strip())
    df["date"] = df['date'].apply(lambda x: x[1:-1])
    df["method"] = df['cols'].apply(lambda x: x[5].strip() if len(x) > 5 else '')
    df["method"] = df['method'].apply(lambda x: x[1:])
    df["endpoint"] = df['cols'].apply(lambda x: x[6].strip() if len(x) > 6 else '')
    df["protocol"] = df['cols'].apply(lambda x: x[7].strip() if len(x) > 7 else '')
    df["protocol"] = df['protocol'].apply(lambda x: x[:-1])
    df["response_code"] = df['cols'].apply(lambda x: try_int(x[8].strip()) if len(x) > 8 else -1)
    df["content_size"] = df['cols'].apply(lambda x: x[9].strip() if len(x) > 9 else '')
    df["content_size"] = df['content_size'].apply(lambda x: 0 if x == '-' else try_int(x))
    df = df[df.endpoint.str.len() > 0]
    df["endpoint"] = df['endpoint'].apply(
        lambda x: RandomizeEndpointUDF(x))  # todo: currently we can not pass python functions around...
    return df


@hipy.compiled_function
def run_logs():
    _pd.set_option('display.expand_frame_repr', False)
    data = gen_data()
    print(logs(data))


def test_logs():
    check_prints(run_logs, """
                                                line                                               cols            ip client_id user_id                                           endpoint                        date method  protocol  response_code  content_size
0  83.149.9.216 - - [17/May/2015:10:05:03 +0000] ...  [83.149.9.216, -, -, [17/May/2015:10:05:03, +0...  83.149.9.216         -       -  /presentations/logstash-monitorama-2013/images...  17/May/2015:10:05:03 +0000    GET  HTTP/1.1            200        203023
""")
