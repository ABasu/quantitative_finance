#!/usr/bin/env python3
# encoding=utf8

"""
"""
import argparse, logging, os, re, signal, sys, time
import pandas as pd

import alphavantage as av

# Parser for command line parameters
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-q', '--queries_per_day', default=450, help='Maximum number of queries allowed per day', type=int)
parser.add_argument('-pq', '--pause_between_queries', default=-1, help='If this is set, we run -q queries and stop', type=int)
parser.add_argument('-tl', '--ticker_list', default='../../data/sac_tickers_largecap.csv,../../data/sac_tickers_midcap.csv,../../data/av_listed_tickers_etfs.csv', help='Comma separated list of CSV stock tickers where first column is ticker (header row is ignored).', type=str)
parser.add_argument('-o', '--output_directory', default='~/Desktop/Server/market_data/', help='Output directory', type=str)
parser.add_argument('-lf', '--log_file', default='_logfile.log', help='Log file. Written to output directory/API_call/.', type=str)
parser.add_argument('-k', '--alphavantage_api_key', default='apikey.cfg', help='The file with the API key used to retrieve data from alphavantage', type=str)
parser.add_argument('-api', '--alphavantage_api_call', default='TIME_SERIES_DAILY', help='The API call used to retrieve data from alphavantage', type=str)
parser.add_argument('-f', '--first_ticker', default=None, help='The first ticker in the list to start with.', type=str)
parser.add_argument('-p', '--parameters', default=[], help='Key=Value (no space) pairs to be passed to API call.', metavar='KEY=VALUE', nargs='+')

args=parser.parse_args()

# Print out a summary of parameters
print('Running with the following parameters:')
for arg in vars(args):
    print('{:>40} : {}'.format(arg, vars(args)[arg]))

# Parse CLI srgs into variables
if args.pause_between_queries < 0:
    pause_between_queries = int((24*60*60) / args.queries_per_day)
    max_queries = None
else:
    max_queries = args.queries_per_day
    pause_between_queries = args.pause_between_queries
ticker_list = [os.path.expanduser(p) for p in args.ticker_list.split(',')]
output_directory = os.path.expanduser(args.output_directory)
av_api_key = args.alphavantage_api_key
av_api_call = args.alphavantage_api_call
log_file = os.path.join(output_directory, av_api_call, args.log_file)
first_ticker = args.first_ticker
current_ticker = None
# Parse parameters into integer, float, or string
parameters = {p.split('=')[0]:p.split('=')[1] for p in args.parameters}
for k in parameters:
    try:
        parameters[k] = int(parameters[k])
    except ValueError:
        try:
            parameters[k] = float(parameters[k])
        except ValueError:
            pass

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
def parse_ticker_list(ticker_list):
    """
    Read a list of CSV files and parse out the first column as a list of ticker

    Parameters
    ----------
    ticker_list : list of strings
        The filenames of CSVs.

    Returns
    -------
    tickers : list of strings
        Parsed tickers.

    """
    tickers_all = []
    for tl in ticker_list:
        tickers = open(tl, 'r').readlines()
        tickers = [l.split(',')[0] for l in tickers[1:]]
        # Remove special classes of tickers that have $ signs
        tickers = [l for l in tickers if not re.search('\$', l)]
        # Change any dots to - for the API
        tickers = [l.replace('.', '-') for l in tickers]
        tickers_all.extend(tickers)
        
    return tickers_all

def parse_log_last_ticker():
    """
    At startup, this function is called to find out what the last ticker was 
    that was being retrieved when last shut down. It reads to logfile to see
    if there's a "Quitting" statement with a ticker name.'

    Returns
    -------
    last_ticker : string or None
        The last ticker, if found, otherwise None.

    """
    try:
        log_last_line = open(log_file, 'r').readlines()[-1]
    except:
        log_last_line = ''
    match_ticker = re.search('\d\d-\d\d-\d\d\s+\d\d:\d\d:\d\d\s+([A-Z]+): Quitting', log_last_line)
    last_ticker = match_ticker.group(1) if match_ticker != None else None
    return last_ticker

def exit_gracefully(sig, frame):
    """
    Logs the ticker about to be retreived, but incomplete, when exiting.

    Parameters
    ----------
    sig : 
    frame : 
        
    Returns
    -------
    None.

    """
    logging.error('{}: Quitting before retrieving data'.format(current_ticker))
    sys.exit(0)

def get_path(ticker):
    """
    Figures out the path for the output file based on the current api call and
    the ticker. 
    NOTE: Does only CSV files for now.

    Parameters
    ----------
    ticker : string
        Stock ticker.

    Returns
    -------
    TYPE: string
        File path.

    """
    return os.path.join(output_directory,av_api_call,'{}.csv'.format(ticker))

def get_ticker_data_from_api(ticker, av_api_call, api_key, parameters={}):
    """
    Queries the API and returns a Dataframe, Dictionary or None

    Parameters
    ----------
    ticker : string
        Stock ticker.

    Returns
    -------
    timeseries : Dataframe, Dictionary, or None
        Parsed data - Dataframe for timeseries, Dictionary for key-value 
        fundamental data, none if failed.
    """
    av_response = av.call_api(ticker, av_api_call, api_key, parameters=parameters)

    return av_response
    
def check_overwrite_tickerfile(ticker, ticker_df):
    ticker_file = get_path(ticker)
    if os.path.exists(ticker_file):
        logging.info('{:>6}: Reading {}.'.format(ticker, ticker_file))
        # File is either CSV (pandas) or JSON
        if ticker_file.lower().split('.')[-1] == 'csv':
            ticker_df_old = pd.read_csv(ticker_file)
            if len(ticker_df) > len(ticker_df_old):
                logging.info('{:>6}: Updating {}.'.format(ticker, ticker_file))
                ticker_df.to_csv(ticker_file, float_format='%.2f')
            else:
                logging.info('{:>6}: {} already up to date.'.format(ticker, ticker_file))
        else:
            # JSON handler goes here
            pass
    else:
        logging.info('{:>6}: Creating file {}.'.format(ticker, ticker_file))
        ticker_df.to_csv(ticker_file, float_format='%.2f')
    

def pause(ticker, n, width=100, segments=100):
    """
    Waits for pause_between_queries while updating the screen with a timer bar

    Parameters
    ----------
    ticker: string
        Current ticker
    n: int
        Index of current ticker
    width : int
        Width of the timer bar. The default is 100.
    segments : int, optional
        Granularity of the timer - i.e. how many times the total weight time is
        divided. The default is 100.

    Returns
    -------
    None.

    """
    for t in range(segments):
        print('{:>6s}: [{val:<{WIDTH}s}] {SECS:>4}s [Ticker {N:} of {LEN:}]'.format(ticker, val='-'*int(width*(t+1)/segments), WIDTH=width, SECS=int(pause_between_queries-((pause_between_queries/segments)*t)), N=n+1, LEN=len(tickers)), end='\r')
        time.sleep(pause_between_queries/segments)
    print(' '*(width+50), end='\r')
    
##############################################################################

signal.signal(signal.SIGINT, exit_gracefully) # Catch keyboard interrupt
tickers = parse_ticker_list(ticker_list)      # Get tickers
first_ticker = parse_log_last_ticker()         # Get last / first ticker

api_key = av.get_api_key(config_file=av_api_key)
n_queries = 0

# Loop until keyboard interrupt
while True:
    for n, ticker in enumerate(tickers):
        if max_queries != None and first_ticker == None:
            logging.info('{} of {} daily queries.'.format(n_queries, max_queries))
            if n_queries >= max_queries:
                logging.error('{}: Quitting before retrieving data - max queries reached.'.format(current_ticker))
                break
        # If this is the first instance of the loop then skip forward to the 
        # last ticker that was about to be processed before quitting.
        if first_ticker:
            if ticker != first_ticker:
                continue                      # Ignore until we see last ticker
            else:
                first_ticker = None            # Delete for future loops
        
        current_ticker = ticker
        pause(ticker, n)
        ticker_df, nq = get_ticker_data_from_api(ticker, av_api_call, api_key, parameters=parameters)

        n_queries += nq
        if isinstance(ticker_df, pd.DataFrame):
            check_overwrite_tickerfile(ticker, ticker_df)
        else:
            continue
                    
    