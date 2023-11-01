#!/usr/bin/env python
# encoding=utf8
"""
A set of functions to work with financial time-series
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logging.basicConfig(format = '%(asctime)-25s %(message)s', level = logging.INFO)

##############################################################################
def interest_rate(a, p, da=None, dp=None):
    """
    Computes the rate of interest. If starting and ending dates are supplied,
    returns annualized compound interest rate. Otherwise, returns simple
    percentage increase/decrese of amount over principal.
    
    Parameters
    ----------
    a : numeric
        the final amount.
    p : numeric
        the starting pricipal.
    da : datestring or pd.Timestamp, optional
        the end date for amount. The default is None.
    dp : datestring or pd.Timestamp, optional
        the start date for principal. The default is None.

    Returns
    -------
    r : float
        the rate of interest.

    """
    
    if da == None or dp == None:
        r = pct(a, p)
    else:
        t = (pd.Timestamp(da)-pd.Timestamp(dp)).days/365
        r = (np.power((a/p), 1/t)-1) * 100
    return r

##############################################################################
def pct(n, base, days=0, years=0):
    """
    Returns n as a percentage rise or fall from base. If either days or years or
    both are supplied an annualized rate will be returned.

    Parameters
    ----------
    n : numeric
        Number to compute percentage rise or fall for
    base : numeric
        Base number to calculate percentage against.

    Returns
    -------
    Float
        Percentage rise or fall.
    """
    if days != 0:
        years += days/365

        
    if years != 0:
        pct = (((n/base)**(1/years))-1) * 100
    else:
        pct = ((n-base)*100)/base
    return pct

##############################################################################
def rolling_avg(series, window, func=np.average):
    """
    Takes a time list of values and computes rolling averages over a window
    sized span. The default is to compute rolling average but any function that 
    takes a list can be applied -- e.g. np.std

    Parameters
    ----------
    series : iterable
        timeseries.
    window : integer
        span to average over.

    Returns
    -------
    Averaged list.

    """
    first = int(window/2)
    second = window - first - 1  
    first = [func(series[:p+1]) for p in range(first)]
    second = [func(series[-p-1:]) for p in range(second)]
    smoothed_series = first + [func(series[start:start+window]) for start in range(len(series)-window+1)] + second
    
    return smoothed_series





##############################################################################
# Not using the following drafts
##############################################################################
def timeseries_match(series, target):
    """
    Takes two time series and returns values from series that are 
    closest to those of target. For each point in the target series 
    only the closest next lower point is retained. Use to, for example,
    convert daily charts to weekly.
    
    Parameters
    ----------
    series : List of timestamps
    target : List of timestamps
    
    Returns
    -------
    target_series : A list of timestamps
    """
    series = sorted(series)
    target = sorted(target)
    target_series = []
    for t in target:
        while series[0] <= t:
            current = series.pop(0)
            if len(series) == 0:
                break
        if current != None:
            target_series.append(current)
        current = None

    return target_series

##############################################################################
def datestring_to_timestamp(datestring, formatstring="%Y-%m-%d"):
    """
    Converts a formatted datestring to a UTC timestamp 
    
    Parameters
    ----------
    datestring : string
        A string parseable to a date.
    formatstring : string
        Date format. The default is "%Y-%m-%d".

    Returns
    -------
    timestamp : timestamp
    """
    date = datetime.strptime(datestring, formatstring)
    timestamp = date.replace(tzinfo=timezone.utc).timestamp()
    
    return timestamp

##############################################################################
def dt64_to_timestamp(dt64):
    """
    Convert numpy UTC dt64 to integer timestamp

    Parameters
    ----------
    dt64 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return int(dt64.astype('datetime64[s]').astype('int'))

##############################################################################
def timestamp_to_utcdate(timestamp):
    """
    Takes a timestamp and generates a date string.

    Parameters
    ----------
    timestamp : timestamp
        A timestamp.
    
    Returns
    -------
    date : TYPE
        A datetime object.

    """
    date = datetime.utcfromtimestamp(timestamp)
    
    return date
    
##############################################################################
def get_yearly_dates(year, freq='W-FRI'):
    """
    Get all dates for a year given a pattern. For example, all 
    Fridays for 2020.
    
    Parameters
    ----------
    year : int
        Year.
    freq : Format, optional
        Freq is a pandas timeseries offset: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    D for daily, W-FRI for fridays. The default is 'W-FRI'.

    Returns
    -------
    dates : list
        A list of datestrings.

    """
    dates = pd.date_range(start=str(year), end=str(year+1), \
                          freq=freq).strftime('%Y-%m-%d').tolist()
    return dates
 
