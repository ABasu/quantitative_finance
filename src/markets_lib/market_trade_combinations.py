#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 16:12:12 2022

@author: anupam
"""

import argparse, logging, os, sys
import pandas as pd

import timeseries as ts


# Parser for command line parameters
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-t', '--tickers', default='../../data/sac_tickers_largecap.csv', help='CSV file with Symbol column containing tickers', type=str)
parser.add_argument('-m', '--market', default='SPY', help='Market ticker', type=str)
parser.add_argument('-o', '--output_directory', default='~/Desktop/Server/market_data/TradeTicker', help='Output directory', type=str)
parser.add_argument('-of', '--output_file', default='trade_ticker.csv', help='Output file', type=str)
parser.add_argument('-lf', '--log_file', default='_logfile.log', help='Log file. Written to output directory', type=str)
parser.add_argument('-a', '--average', default='True', help='Average trades for a single run?', type=bool)
parser.add_argument('-mp', '--min_profit', default='2,5,10,15', help='Minimum Profit', type=str)
parser.add_argument('-ml', '--max_loss', default='10,25,50', help='Maximum Loss', type=str)
parser.add_argument('-wl', '--window_long', default='180,365,730', help='Window Long', type=str)
parser.add_argument('-ws', '--window_short', default='1,7,30', help='Window Short', type=str)
args=parser.parse_args()

# Print out a summary of parameters
print('Running with the following parameters:')
for arg in vars(args):
    print('{:>40} : {}'.format(arg, vars(args)[arg]))

# Parse CLI srgs into variables
tickers = os.path.expanduser(args.tickers)
market = os.path.expanduser(args.market) 
output_directory = os.path.expanduser(args.output_directory)
output_file = os.path.join(output_directory, args.output_file)
log_file = os.path.join(output_directory, args.log_file)
average = args.average

mp = [int(a) for a in args.min_profit.split(',')]
ml = [int(a) for a in args.max_loss.split(',')]
wl = [int(a) for a in args.window_long.split(',')]
ws = [int(a) for a in args.window_short.split(',')]

# Set up logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

fh = logging.FileHandler(log_file)
fh.setLevel(logging.ERROR)
sh = logging.StreamHandler(sys.stderr)
sh.setLevel(logging.INFO)
fmt = '%(asctime)s %(message)s'
dfmt = '%y-%m-%d  %H:%M:%S'
logging.basicConfig(handlers=(fh, sh), format=fmt, datefmt=dfmt, level=logging.INFO)

##############################################################################
try:
    csv = pd.read_csv(tickers)
    tickers = list(csv['Symbol'])
except FileNotFoundError:
    # if not a file, try treating it as csv list of tickers
    logging.info('File {} not found. Trying as CSV of tickers'.format(tickers))
    tickers = tickers.split(',')

TM = ts.TradeMarket(tickers, market, min_profit=mp, max_loss=ml, window_long=wl, window_short=ws)
TM.trade(print_report=True, average=average)
TM.report_table_save(filename=output_file)


