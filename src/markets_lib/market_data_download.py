#!/usr/bin/env python3
# encoding=utf8

"""
"""
import argparse, datetime, logging, os, re, signal, sys, time
import pandas as pd

import alphavantage as av
import timeseries as ts


# Parser for command line parameters
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-q', '--queries_per_day', default=450, help='Maximum number of queries allowed per day', type=int)
parser.add_argument('-tl', '--ticker_list', default='../../data/sac_tickers_largecap.csv,../../data/sac_tickers_midcap.csv,../../data/av_listed_tickers_etfs.csv', help='Comma separated list of CSV stock tickers where first column is ticker (header row is ignored).', type=str)
parser.add_argument('-o', '--output_directory', default='~/Desktop/Server/market_data/', help='Output directory', type=str)
parser.add_argument('-lf', '--log_file', default='_logfile.log', help='Log file. Written to output directory/API_call/.', type=str)
parser.add_argument('-api', '--alphavantage_api_call', default='TIME_SERIES_DAILY', help='The API call used to retrieve data from alphavantage', type=str)
parser.add_argument('-f', '--first_ticker', default=None, help='The first ticker in the list to start with.', type=str)
parser.add_argument('-i', '--interval', default='5min', help='For intraday data: api TIME_SERIES_INTRADAY_EXTENDED', type=str)
parser.add_argument('-ipm', '--ignore_prepost_market', default=True, help='For intraday data: api TIME_SERIES_INTRADAY_EXTENDED', type=bool)
parser.add_argument('-mo', '--mkt_open', default='9:30', help='For intraday data: api TIME_SERIES_INTRADAY_EXTENDED', type=str)
parser.add_argument('-mc', '--mkt_close', default='16:00', help='For intraday data: api TIME_SERIES_INTRADAY_EXTENDED', type=str)
args=parser.parse_args()

# Print out a summary of parameters
print('Running with the following parameters:')
for arg in vars(args):
    print('{:>40} : {}'.format(arg, vars(args)[arg]))

# Parse CLI srgs into variables
pause_between_queries = int((24*60*60) / args.queries_per_day)
ticker_list = [os.path.expanduser(p) for p in args.ticker_list.split(',')]
output_directory = os.path.expanduser(args.output_directory)
av_api_call = args.alphavantage_api_call
log_file = os.path.join(output_directory, av_api_call, args.log_file)
first_ticker = args.first_ticker
current_ticker = None
interval = args.interval
ignore_prepost_market = args.ignore_prepost_market
mkt_open = tuple(int(x) for x in args.mkt_open.split(':'))
mkt_open = datetime.time(mkt_open[0], mkt_open[1])
mkt_close = tuple(int(x) for x in args.mkt_close.split(':'))
mkt_close = datetime.time(mkt_close[0], mkt_close[1])

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

def get_ticker_data_from_api(ticker, parameters={'outputsize':'full'}):
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
    av_response = av.call_api(symbol=ticker, function=av_api_call, parameters=parameters)
    parsed_response, columns = av.parse_response(av_response, function=av_api_call, symbol=ticker)
    if parsed_response:
        if isinstance(parsed_response, list):
            data_frame = ts.timeseries_df(parsed_response, columns)
            if av_api_call == 'TIME_SERIES_INTRADAY_EXTENDED' and ignore_prepost_market == True:
                mask = data_frame['Date'].apply(lambda d: d.time() < mkt_open or d.time() > mkt_close)
                data_frame.drop(data_frame[mask].index)
            data_frame = data_frame.sort_values('Date')            
            return data_frame
        else:
            dictionary = parsed_response
            return dictionary
    else:
        return None
    
def check_overwrite_tickerfile(ticker, ticker_data):
    ticker_file = get_path(ticker)
    if os.path.exists(ticker_file):
        logging.info('{:>6}: Reading {}.'.format(ticker, ticker_file))
        # File is either CSV (pandas) or JSON
        if ticker_file.lower().split('.')[-1] == 'csv':
            ticker_df_old = pd.read_csv(ticker_file, index_col=[0], parse_dates=['Date'])
            if len(ticker_data) > len(ticker_df_old):
                logging.info('{:>6}: Updating {}.'.format(ticker, ticker_file))
                ticker_data.to_csv(ticker_file, float_format='%.2f')
            else:
                logging.info('{:>6}: {} already up to date.'.format(ticker, ticker_file))
        else:
            # JSON handler goes here
            pass
    else:
        logging.info('{:>6}: Creating file {}.'.format(ticker, ticker_file))
        ticker_data.to_csv(ticker_file, float_format='%.2f')
    

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

# Loop until keyboard interrupt
while True:
    for n, ticker in enumerate(tickers):
        # If this is the first instance of the loop then skip forward to the 
        # last ticker that was about to be processed before quitting.
        if first_ticker:
            if ticker != first_ticker:
                continue                      # Ignore until we see last ticker
            else:
                first_ticker = None            # Delete for future loops
        
        current_ticker = ticker
        pause(ticker, n)
        if av_api_call == 'TIME_SERIES_INTRADAY_EXTENDED':
            parameters = {'interval':interval, 'adjusted':'true', 'pause_between_queries':pause_between_queries}
        else:
            parameters = {'outputsize':'full'}
        ticker_data = get_ticker_data_from_api(ticker, parameters=parameters)
        if isinstance(ticker_data, pd.DataFrame):
            check_overwrite_tickerfile(ticker, ticker_data)
        else:
            continue
                    
    