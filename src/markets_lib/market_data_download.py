#!/usr/bin/env python3
# encoding=utf8

help_string = """
Command-line program to download data from the Alphavantage API. Allows setting
queries_per_day in which case it sends queries continuously at intervals and keeps
running. Otherwise, pause_between_queries can be set and the program runs a maximum
of --queries_per_day queries and stops. Tickers to be downloaded are gathered from 
a list of CSV files - default is about 3000 tickers. Look at the alphavantage 
documentation at https://www.alphavantage.co/documentation/ for parameters needed 
for each function call. Parameters are supplied as KEY=VALUE pairs without spaces. 
The program can be stopped with a keyboard interrupt and it writes the last ticker 
to a log file before exiting. On startup this ticker is read from the log and 
parsing starts here. first_ticker can also be set manually. 
"""
import argparse, logging, os, re, signal, sys, time
import pandas as pd

import alphavantage as av

# Parser for command line parameters
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description = help_string)
parser.add_argument('-q',  '--queries_per_day', default=100, help='Maximum number of queries allowed per day', type=int)
parser.add_argument('-pq', '--pause_between_queries', default=-1, help='If this is set, we run -q queries and stop', type=int)
parser.add_argument('-tl', '--ticker_list', default='../../data/sac_tickers_largecap.csv,../../data/sac_tickers_midcap.csv', help='Comma separated list of CSV stock tickers where first column is ticker (header row is ignored).', type=str)
parser.add_argument('-e',  '--exchange', default=None, help='Exchange code. Default is NYSE (i.e. empty)', type=str)
parser.add_argument('-o',  '--output_directory', default='~/Desktop/Server/market_data/', help='Output directory', type=str)
parser.add_argument('-k',  '--alphavantage_api_key', default='apikey.cfg', help='The file with the API key used to retrieve data from alphavantage', type=str)
parser.add_argument('-af', '--alphavantage_api_function', default='TIME_SERIES_DAILY', help='The API call used to retrieve data from alphavantage', type=str)
parser.add_argument('-l',  '--log_file', default='_logfile.log', help='Log file. Written to output directory/API_call/.', type=str)
parser.add_argument('-f',  '--first_ticker', default=None, help='The first ticker in the list to start with.', type=str)
parser.add_argument('-p',  '--parameters', default=[], help='Key=Value (no space) pairs to be passed to API call.', metavar='KEY=VALUE', nargs='+')

args=parser.parse_args()

# Print out a summary of parameters
print('Running with the following parameters:')
for arg in vars(args):
    print('{:>40} : {}'.format(arg, vars(args)[arg]))

# Parse CLI args into variables
# Are we running in max_queries and stop or continues background mode?
if args.pause_between_queries < 0:
    max_queries = None
    pause_between_queries = int((24*60*60) / args.queries_per_day)
else:
    max_queries = args.queries_per_day
    pause_between_queries = args.pause_between_queries
ticker_list = [os.path.expanduser(p) for p in args.ticker_list.split(',')]
exchange = args.exchange 
output_directory = os.path.expanduser(args.output_directory)
av_api_key = args.alphavantage_api_key
av_api_function = args.alphavantage_api_function
log_file = args.log_file
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

##############################################################################
def check_overwrite_tickerfile(ticker, api_function, exchange, ticker_df):
    """
    For a given dataframe, checks to see if the data already exists on disk. If
    so, loads the file and compares to see if there are new lines in the dataframe.
    In that case, the new data is written over the existing file.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    ticker_df : pd.DataFrame
        Retreived dataframe from API.

    Returns
    -------
    None.

    """
    ticker_file = get_path(ticker, api_function, exchange)
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
    
##############################################################################
def exit_gracefully():
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
    logging.error('{}: Quitting before retrieving data'.format(ticker))
    sys.exit(0)

##############################################################################
def get_dir(api_function, exchange):
    """
    Return the directory for the current call. e.g. TIME_SERIES_DAILY or 
    TIME_SERIES_DAILY.BSE etc

    Parameters
    ----------
    ticker : str
        Ticker.
    exchange : str
        Exchange code.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return '.'.join([api_function, exchange]) if exchange != None else api_function

##############################################################################
def get_path(ticker, api_function, exchange):
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
    return os.path.join(output_directory, get_dir(api_function, exchange),'{}.csv'.format(ticker))

##############################################################################
def parse_log_last_ticker(log_file):
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
    match_ticker = re.search('\d\d-\d\d-\d\d\s+\d\d:\d\d:\d\d\s+([A-Z\d]+): Quitting', log_last_line)
    last_ticker = match_ticker.group(1) if match_ticker != None else None

    return last_ticker

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
        # remove commented lines
        tickers = [t for t in tickers if t[0]!='#']
        tickers = [l.split(',')[0] for l in tickers[1:]]
        # Remove special classes of tickers that have $ signs
        tickers = [l for l in tickers if not re.search('\$', l)]
        # Change any dots to - for the API
        tickers = [l.replace('.', '-') for l in tickers]
        tickers_all.extend(tickers)
        
    return tickers_all

##############################################################################
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

if __name__ == "__main__":
    signal.signal(signal.SIGINT, exit_gracefully) # Catch keyboard interrupt

    # Set up logging
    log_file = os.path.join(output_directory, get_dir(av_api_function, exchange), log_file)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.ERROR)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    fmt = '%(asctime)s %(message)s'
    dfmt = '%y-%m-%d  %H:%M:%S'
    logging.basicConfig(handlers=(fh, sh), format=fmt, datefmt=dfmt, level=logging.INFO)

    tickers = parse_ticker_list(ticker_list)      # Get tickers
    if first_ticker == None:
        first_ticker = parse_log_last_ticker(log_file)    # Get last / first ticker
    
    api_key = av.get_api_key(config_file=av_api_key)
    n_queries = 0

    # Loop until keyboard interrupt
    while True:
        # Loop thru complete list of tickers
        for n, ticker in enumerate(tickers):
            # If this is the first instance of the loop then skip forward to the 
            # last ticker that was about to be processed before quitting.
            if first_ticker:
                if ticker != first_ticker:
                    continue                      # Ignore until we see last ticker
                else:
                    first_ticker = None            # Delete for future loops

            # We don't go over max_queries if it's set
            if max_queries != None:
                logging.info('{} of {} daily queries done.'.format(n_queries, max_queries))
                if n_queries >= max_queries:
                    logging.error('{}: Quitting before retrieving data - max queries reached.'.format(ticker))
                    break
            
            pause(ticker, n)
            ticker_data, nq = av.call_api(ticker, av_api_function, api_key, \
                                exchange=exchange, parameters=parameters, normalize_column_headings=True)
    
            n_queries += nq
            if isinstance(ticker_data, pd.DataFrame):
                check_overwrite_tickerfile(ticker, av_api_function, exchange, ticker_data)
            else:
                continue
                        
        