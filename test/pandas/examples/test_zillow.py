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
def extractBd(val):
    max_idx = val.find(' bd')
    if max_idx < 0:
        max_idx = len(val)
    s = val[:max_idx]

    # find comma before
    split_idx = s.rfind(',')
    if split_idx < 0:
        split_idx = 0
    else:
        split_idx += 2
    r = s[split_idx:]
    return int(r)


@hipy.compiled_function
def extractType(title):
    t = title.lower()
    type = 'unknown'
    if 'condo' in t or 'apartment' in t:
        type = 'condo'
    if 'house' in t:
        type = 'house'
    return type


@hipy.compiled_function
def extractBa(val):
    max_idx = val.find(' ba')
    if max_idx < 0:
        max_idx = len(val)
    s = val[:max_idx]
    # find comma before
    split_idx = s.rfind(',')
    if split_idx < 0:
        split_idx = 0
    else:
        split_idx += 2
    r = s[split_idx:]
    return int(r)


@hipy.compiled_function
def extractSqft(val):
    max_idx = val.find(' sqft')
    if max_idx < 0:
        max_idx = len(val)
    s = val[:max_idx]

    split_idx = s.rfind('ba ,')
    if split_idx < 0:
        split_idx = 0
    else:
        split_idx += 5
    r = s[split_idx:]
    r = r.replace(',', '')
    return int(r)


@hipy.compiled_function
def extractOffer(offer):
    if 'Sale' in offer:
        offer = 'sale'
    if 'Rent' in offer:
        offer = 'rent'
    if 'SOLD' in offer:
        offer = 'sold'
    if 'foreclose' in offer.lower():
        offer = 'foreclosed'

    return offer


@hipy.compiled_function
def extractPrice(x):
    price, offer, val, sqft = x['price'], x['offer'], x['facts and features'], x['sqft']

    if offer == 'sold':
        # price is to be calculated using price/sqft * sqft
        s = val[val.find('Price/sqft:') + len('Price/sqft:') + 1:]
        r = s[s.find('$') + 1:s.find(', ') - 1]
        price_per_sqft = int(r)
        price = price_per_sqft * sqft
    elif offer == 'rent':
        max_idx = price.rfind('/')
        price = int(price[1:max_idx].replace(',', ''))
    else:
        # take price from price column
        price = int(price[1:].replace(',', ''))

    return price


@hipy.compiled_function
def gen_data():
    return pd.DataFrame.from_dict(
        {
            "facts and features": ["2 bds , 1 ba , 1,000 sqft", "3 bds , 2 ba , 1,500 sqft",
                                   "4 bds , 3 ba , 2,000 sqft"],
            "title": ["House For Sale", "House For Rent", "House For Sale"],
            "price": ["$100,000", "$2,000/mo", "$200,000"],
            "postal_code": ["10001", "10002", "10003"],
            "city": ["New York", "New York", "New York"],
            "state": ["NY", "NY", "NY"],
            "url": ["https://www.zillow.com/homedetails/1-Fifth-Ave-APT-10B-New-York-NY-10003/31541718_zpid/",
                    "https://www.zillow.com/homedetails/1-Fifth-Ave-APT-10B-New-York-NY-10003/31541718_zpid/",
                    "https://www.zillow.com/homedetails/1-Fifth-Ave-APT-10B-New-York-NY-10003/31541718_zpid/"],
            "address": ["1 Fifth Ave APT 10B", "1 Fifth Ave APT 10B", "1 Fifth Ave APT 10B"],
        }
    )

def format_zipcode(zc) -> str:
    return '%05d' % int(zc)
@hipy.compiled_function
def zillow(df):
    df['bedrooms'] = df['facts and features'].apply(extractBd)
    df = df[df['bedrooms'] < 10]
    df['type'] = df['title'].apply(extractType)
    df = df[df['type'] == 'house']

    df['zipcode'] = df['postal_code'].apply(format_zipcode)

    df['city'] = df['city'].apply(lambda x: x[0].upper() + x[1:].lower())
    df['bathrooms'] = df['facts and features'].apply(extractBa)
    df['sqft'] = df['facts and features'].apply(extractSqft)
    df['offer'] = df['title'].apply(extractOffer)
    df['price'] = df[['price', 'offer', 'facts and features', 'sqft']].apply(extractPrice, axis=1)

    df = df[(100000 < df['price']) & (df['price'] < 2e7)]

    df = df[["url", "zipcode", "address", "city", "state",
             "bedrooms", "bathrooms", "sqft", "offer", "type", "price"]]
    return df


@hipy.compiled_function
def run_zillow():
    _pd.set_option('display.expand_frame_repr', False)
    data = gen_data()
    print(zillow(data))


def test_zillow():
    check_prints(run_zillow, """
                                                 url zipcode              address      city state  bedrooms  bathrooms  sqft offer   type   price
2  https://www.zillow.com/homedetails/1-Fifth-Ave...   10003  1 Fifth Ave APT 10B  New york    NY         4          3  2000  sale  house  200000
""")
