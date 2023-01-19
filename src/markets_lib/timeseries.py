#!/usr/bin/env python
# encoding=utf8
"""
A set of functions to work with financial time-series

Dataframes have the following attributes:
    Price_Type: 'Unadjusted', 'Adjusted'
    Pct_Window: integer
    Splits: list of (date, ratio1, ratio2) tuples
"""

import datetime, logging, operator, os, re
import pandas as pd
import numpy as np
import seaborn as sns

#from . import stats    # For Jupyter notebooks
import stats           # For everything else

logging.basicConfig(format = '%(funcName)s %(asctime)-25s %(message)s', level = logging.INFO)
 
##############################################################################
class Ticker(object):
    def __init__(self, ticker=None, timeseries=None, rootdir='~/Desktop/Server/market_data/', av_func='TIME_SERIES_DAILY'):
        """
        Initializes a ticker-timeseries object.

        Parameters
        ----------
        ticker : str, optional
            A stock ticker. The default is None.
        timeseries : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        self.

        """
        self.timeseries = None
        self.ticker = ticker.upper()
        self.column = None      # The column to focus on for single column 
                                # operations like high_low
        
        # These properties of the overall timeseries will be passed to a new
        # instance (subset etc) from the current timeseries
        self.price_type = None
        self.splits = None

        # These properties would not be passed, they would need to be 
        # recalculated for a new subset
        self.beta = None
        
        if ticker != None:
            self.read_csv(ticker, rootdir, av_func)
        elif isinstance(timeseries, pd.DataFrame):
            self.timeseries = timeseries.timeseries
            self.ticker = timeseries.ticker
            self.price_type = timeseries.price_type
            self.splits = timeseries.splits

    def read_csv(self, ticker, rootdir, \
                 av_func, adjustsplits=True):
        """
        Reads a csv timeseries and optionally calls adjustsplits on it. Resulting 
        columns are labeled 'Date' and 'Close'
    
        Parameters
        ----------
        ticker : string
            A stock ticker.
        rootdir : string, optional
            Location of datafiles. The default is '~/Desktop/Server/market_data/'.
        av_func : string, optional
            The timeseries function - serves as subdirectory name. The default is 'TIME_SERIES_DAILY'.
        adjustsplits : Boolean, optional
            Whether to adjust for stock splits. The default is True.
    
        Returns
        -------
        self
        """
        self.ticker = ticker
        ticker_file = os.path.join(os.path.expanduser(rootdir), av_func, ticker.upper()+'.csv')
        try:
            timeseries = pd.read_csv(ticker_file, index_col=[0], parse_dates=['Date'])
            timeseries.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            timeseries.reset_index(drop=True)
            self.timeseries = timeseries
            self.column = 'Close'
            self.price_type = 'Unadjusted'
        except FileNotFoundError:
            logging.error('read_csv(): File not found: {}'.format(ticker_file))
            return None
        if adjustsplits:
            self.adjust_splits()
        return self
    
    def adjust_splits(self):
        """
        Takes a timeseries dataframe where the first column is time stamp 
        and detects stock splits based on self.column (100% or more rise 
        or fall over a day).
    
        Returns
        -------
        sets timeseries dataframe where the first column is time stamp.
        Sets Splits attribute on df. Returns self.
    
        """
        splits = []
        factors = np.ones((len(self.timeseries)))
        for i in range(len(self.timeseries)-1, 0, -1):
            # Protect against div by zero where stock price is 0
            if self.timeseries.loc[i, self.column] != 0:     
                cur = self.timeseries.loc[i, self.column]
                prev = self.timeseries.loc[i-1, self.column]
                change = round((prev-cur) / min(cur, prev))
            else:
                change = 0
                
            if change > 0:
                logging.info('adjust_splits(): {:>6}: {}:1 {} detected on {}'.format(self.ticker, \
                        change+1, 'split', self.timeseries[self.timeseries.columns[0]].loc[i]))
                splits.append((self.timeseries.loc[i]['Date'], change+1, 1))
                f = [1/(change+1)]*i + [1]*(len(self.timeseries)-i)
                factors = factors * f
            elif change < 0:
                logging.info('adjust_splits(): {:>6}: 1:{} {} detected on {}'.format(self.ticker, \
                        (-1*change)+1, 'reverse split', self.timeseries[self.timeseries.columns[0]].loc[i]))
                splits.append((self.timeseries.loc[i]['Date'], 1, (-1*change)+1))
                f = [1*((-1*change)+1)]*i + [1]*(len(self.timeseries)-i)
                factors = factors * f
       
        # Multiply Open, High, Low, Close by factor, but don't multiply Volume
        for c in range(1, len(self.timeseries.columns)-1):
            self.timeseries[self.timeseries.columns[c]] *= factors
        self.splits = splits
        self.price_type = 'Adjusted'
    
        return self

    def daterange_slice(self, start_date=None, end_date=None, in_place=True):
        """
        Takes start and end dates as strings or pd.Timestamp objects and returns
        a self.timeseries if in_place is True or a new Ticker object if in_place 
        is False

        Parameters
        ----------
        start_date : str or pd.Timestamp, optional (yyyy-mm-dd preferred)
            Start date - beginning of timeseries if None. The default is None.
        end_date : str or pd.Timestamp, optional (yyyy-mm-dd preferred)
            End date - end of timeseries if None. The default is None.
        in_place : bool, optional
            Whether to modify self.timeseries or return new ticker object. 
            The default is True.

        Returns
        -------
        TYPE
            self or Ticker.
        """
        if start_date == None:
            start_date = self.timeseries['Date'][0]
        if end_date == None:
           end_date = self.timeseries['Date'][len(self.timeseries)-1]
        timeseries = self.timeseries[self.timeseries['Date']>=pd.Timestamp(start_date)]
        timeseries = timeseries.reset_index(drop=True)
        timeseries = timeseries[self.timeseries['Date']<=pd.Timestamp(end_date)]
        if in_place:
            self.timeseries = timeseries
            self.beta = None
            self.daily_variance = None
            return self
        else:
            ticker = Ticker()
            ticker.ticker = self.ticker
            ticker.timeseries = timeseries
            ticker.column = self.column
            ticker.price_type = self.price_type
            ticker.splits = self.splits
            return ticker
      
    def drop_columns(self, drop=None, keep=['Date', 'Close']):
        """
        Deletes a set of columns from self.timeseries. If self.column is dropped
        then self.column is set to self.timeseries.columns[1] - i.e. first column
        after 'Date,' normally.

        Parameters
        ----------
        drop : list of strings, optional
            Columns to drop. Generated from keep if drop is None. The default is None.
        keep : list of strings, optional
            Columns to keep. Only used if drop is None. The default is ['Date', 'Close'].

        Returns
        -------
        self

        """
        if drop == None:
            drop = [c for c in self.timeseries.columns if c not in keep]
        if 'Date' in drop:
            logging.warning('drop_columns(): {:>6}: Dropping the Date column'.format(self.ticker))
        df = self.timeseries.drop(columns=drop)
        self.timeseries = df
        if not self.column in self.timeseries.columns:
            logging.warning('drop_columns(): {:>4}: {} was dropped, setting to {}' \
                            .format(self.ticker, self.column, self.timeseries.columns[1]))
            self.column = self.timeseries.columns[1]
        
        return self

    def map_operation(self, ts, operation=operator.sub, column=None):
        """
        Applies a function to column (self.column if column=None) where column
        and ts.timeseries.column are supplied as first and second parameters to 
        operation. e.g. sub(self.column, ts.timeseries.column) should add a new 
        column to self.timeseries with the label in the form 
        sub(ticker, ts.ticker)(column).
        
        If the necessary column of the for pct_nnnn doesn't exist then it is 
        created. In general pct_changed tickers should be subtracted to give
        performance relative to market.
        
        Column MUST EXIST for self. pct_nnnn type columns can be generated on 
        demand for ts.

        Parameters
        ----------
        ts : Ticker
            A ticker object. Usually the market ticker.
        operation : function, optional
            function that takes two parameters and returns one. The default 
            is operator.sub.
        column : str, optional
            Name of column to perform operation on. The default is None.

        Returns
        -------
        self
            Generates a new column in self.timeseries. and sets it as current column.
        """
        if column == None:
            column = self.column
        try:
            op = operation.__name__
        except:
            op = ''
            
        if column not in ts.timeseries.columns:
            logging.warning('map_operation(): {} does not exist for {}. Generating against \'Price\''.format(column, ts.ticker))
            pct_window = int(column[-4:])
            ts.pct_change(window_size=pct_window)
            
        label = '{}({},{})({})'.format(op, self.ticker, ts.ticker, ts.column)
        df = self.timeseries[['Date', column]]
        mdf = ts.timeseries[['Date', column]]
        merged = df.merge(mdf, on=['Date'], how='left')
        merged.columns = ['Date', self.ticker, ts.ticker]
        series = list(map(operator.sub, merged[self.ticker], merged[ts.ticker]))
        self.column = label
        self.timeseries[label] = series

        return self
    
    def pct_change(self, window_size=0, column='Close'):
        """
        Computes percentage change for a window_size span for column. Generates 
        a new column named pct_nnnn where nnnn is the window_size. i.e. convert 
        stock prices to percentage growth based on the starting point or 
        window_size (to get daily change use window_size 1, to use the start of 
        the data as reference, use 0)

        Parameters
        ----------
        window_size: int
        column: string
        
        Returns
        -------
        self
            Prices converted to percentage growth in new column.
        """
        if re.search('{}_pct_{:04}'.format(column, window_size), column):
            logging.warning('pct_change(): {:>6}: pct_change() called on already pct converted column. Returning as is.'.format(self.ticker))
            return self
        col = '{}_pct_{:>04}'.format(column, window_size)
        if col in self.timeseries.columns:
            logging.warning('pct_change(): {:>6}: {} already exists. Returning as is.'.format(self.ticker, col))
            self.column = col
            return self
            
        first = self.timeseries[column].first_valid_index()
        last = self.timeseries[column].last_valid_index()
        nseries = self.timeseries[['Date', column]][first : last+1]
        nseries = nseries.reset_index(drop=True)
        if window_size == 0:
            pct_series = [stats.pct(r[column], nseries.iloc[0][column]) for i, r in nseries.iterrows()]
        else:
            pct_series = [stats.pct(r[column], nseries.iloc[max(i-window_size, 0)][column]) for i, r in nseries.iterrows()]

        self.timeseries[col] = pct_series
        self.column = col
        return self

    def get_daterange_highlow(self, date=None, price=None, days=364, adjust_date=False, \
                          return_bool=True, bool_threshold=2, column=None):
        """
        For a given date and going back a given number of days, returns the % below
        the max and % above the min that the current price is. e.g. to know if the
        current price is at a 52week low, set date to 364. If it is the 52 week, 
        lowest point the difference from the bottom should be 0%. 
        
        If return_bool is set, then a True/False value is returned in the BOTTOM
        is within bool_threshold.
    
        Parameters
        ----------
        date : a date string
            The 'current' date.
        price : numeric, optional
            If you want to ignore the 'current' price and compute low/high relative
            to a given price. The default is None.
        days : int, optional
            The window for getting low/high prices. The default is 364.
        adjust_date : boolean, optional
            If set to true, then the first date available is used in case the date
            range overruns the beginning og the series. If set to False, None is 
            returned in these cases. The default is False.
        return_bool: boolean
            Instead of a tuple of percentage, return a boolean that test only
            for low. Default: True
        bool_threshold: int
            The percentage above the bottom that will return True. Default: 2
        column: str
            The column to operate on.
            
        Returns
        -------
        tuple of floats or bool
            high and low values as % diffs.
        """
        if column == None:
            column = self.column

        if date == None:
            date = self.timeseries.iloc[-1]['Date']
        date = pd.Timestamp(date)
        if price == None:
            price = self.timeseries[self.timeseries['Date']==date][column].iloc[0]
        start_date = self.timeseries[self.timeseries['Date']<=date-pd.Timedelta(days=days)]['Date']
        if len(start_date) == 0:
            if adjust_date:
                start_date = self.timeseries.iloc[0]['Date']
                logging.warning('get_daterange_highlow(): {:>6}: Data begins less than {} days before {}, \
                    using the first day data is available for'.format(self.ticker, days, date))
            else:
                return None
        else:
            start_date = start_date.values[-1]
        
        wdf = self.timeseries[(self.timeseries['Date']>=start_date) & (self.timeseries['Date']<=date)]
        pmax = max(wdf[column])
        pmin = min(wdf[column])
        high, low = stats.pct(price, pmax), stats.pct(price, pmin)
            
        if return_bool:
            # Return high
            if bool_threshold <= 0:
                high_low = False if high < bool_threshold else True
            else:
                high_low = False if low > bool_threshold else True
        else:
            high_low = (high, low)
            
        return high_low

    def get_beta(self, market, window_size=1, column=None):
        """
        Takes a pct_nnnn column in self.timeseries and the same column in market
        and generates a beta score. If a column is supplied, it needs to exist
        in BOTH self and market. If not, window_size is used to generate a new 
        column.

        Parameters
        ----------
        market : TYPE
            DESCRIPTION.
        window_size : TYPE, optional
            DESCRIPTION. The default is 1. i.e. daily
        column : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        orig_column = self.column
        # If a column is supplied, and exists, we use it, otherwise, we use 
        # window_size to generate a col_pct_nnnn column
        if column != None:
            # ... but it doesn't exist
            if column not in self.timeseries.columns:
                logging.warning('get_beta(): {:>6}: {} doesn\'t exist. Using pct_{:>04} from window_size'\
                                .format(self.ticker, column, window_size))
                # ok, so we'll generate a column name from window_size
                column = '{}_pct_{:04}'.format(column, window_size)
        # column name wasn't supplied, generate from window_size
        else:
            column = '{}_pct_{:04}'.format(self.column, window_size)
        
        # Does the column exist. Generate otherwise.
        if column not in self.timeseries.columns:
            self.pct_change(window_size=window_size, column=self.column)
        if column not in market.timeseries.columns:
            market.pct_change(window_size=window_size, column=self.column)
            logging.warning('get_beta(): {:>6}: {} doesn\'t exist. Using pct_{:>04} from window_size'\
                            .format(market.ticker, column, window_size))

        merged = self.timeseries[['Date', column]].merge(market.timeseries[['Date', column]], on=['Date'], how='inner')
        tcol, mcol = (self.ticker+' '+column, market.ticker+' '+column)
        merged.columns = ['Date', tcol, mcol]
        corr = merged[tcol].corr(merged[mcol])
        
        # Restore original column, so generating beta doesn't shift column focus
        self.column = orig_column
        self.beta = corr*(merged[tcol].std()/merged[mcol].std())
        return self.beta

    def get_price(self, date):
        return self.timeseries[self.timeseries['Date']==pd.Timestamp(date)][self.column].values[0]
    

##############################################################################
class TradeTicker(object):
    """
    A trading strategy can consist of the following stages:
        * Buy point
        * Sell point
        
    Buy is triggered when the following condition is met.
        * n_day_low or n_day_low against market
    """

    def __init__(self, ticker, market, \
                 min_profit=5, max_loss=50, \
                 window_long=364, window_short=1, \
                 tax_rates=(15, 30), against_market=True, \
                 stdlimit_window_short=2, stdshift_window_long=True):
        self.ticker = ticker
        self.market = market
        
        # These are our variables
        self.min_profit = min_profit
        self.max_loss = max_loss
        self.window_long = window_long
        self.window_short = window_short
        self.tax_longterm = tax_rates[0]
        self.tax_shortterm = tax_rates[1]
        self.against_market = against_market
        self.stdlimit_window_short = stdlimit_window_short
        
        self.own = False
        self.buy_dates = []
        self.sell_dates = []
        self.buy_price = None
        self.limit = None
        self.trades_list = None
        self.trades_df = None
        
        # Compute ticker parameters that only need to be generated once
        self.beta = self.ticker.get_beta(market, window_size=self.window_short)
        
        # Generate daily pct change. This is set to current column. Note that if
        # beta is generated above, then this column is already generated, this 
        # call will just switch the ticker.column (beta doesn't sitch columns)
        # But we want focus to remain on 'Close'
        self.ticker.pct_change(self.window_short)
        self.window_short_std = self.ticker.timeseries[self.ticker.column].std()
        self.ticker.column = 'Close'
        # We generate long (usually year) growth. This will be used if against 
        # market is True. This column will be current column e.g. 
        # 'sub(MSFT,SPY)(pct_0365)'
        if against_market:
            self.ticker.pct_change(self.window_long)
            self.ticker.map_operation(market)
            if stdshift_window_long:
                logging.info('{}: Shifting column {} so that no negative values or zeroes exist'.format('TradeTicker.init()', self.ticker.column))
                self.window_long_shift = abs(self.ticker.timeseries[self.ticker.column].min())
                self.ticker.timeseries[self.ticker.column] += self.window_long_shift + 1
                
    def trade(self, ignore_unfinished=True):
        for date in self.ticker.timeseries['Date']:
            # If we don't own the stock ...
            if self.buy_price == None:
                # All buy triggers should be in this list. buy is only true 
                # if all triggers return true
                buy = all([self.ticker.get_daterange_highlow(date=date, days=self.window_long)])
                if buy:
                    self.buy_dates.append(date)
                    self.buy_price = self.ticker.timeseries[self.ticker.timeseries['Date']==date]['Close'].values[0]
                    self.limit = self.buy_price - (self.buy_price * (self.max_loss/100))
            # We own the stock ...
            else:
                price = self.ticker.timeseries[self.ticker.timeseries['Date']==date]['Close'].values[0]
                # This triggers sale
                if price <= self.limit:
                    self.sell_dates.append(date)
                    self.buy_price = None
                # Otherwise we check if limit needs to be adjusted
                else:
                    days_owned = (date - self.buy_dates[-1]).days
                    tax_rate = self.tax_longterm if days_owned > 365 else self.tax_shortterm
                    # This is the price we have to hit to make min_profit after tax
                    min_protect_gain_price = ((self.buy_price * (100+self.min_profit)/100) / (100-tax_rate)) * 100
                    # This is the bottom beyond which we assume it's not random 
                    # variation. Once this goes above our min_profit, we start 
                    # setting limits
                    random_variance_margin = price * (self.stdlimit_window_short*self.window_short_std/100)
                    random_variance_bottom = price - random_variance_margin
                    # We adjust the limit is the bottom point of random variance is 
                    # above our min_gain target. But once it is set, it only follows
                    # the ticker up, never comes down, i.e. we protect our gains.
                    if random_variance_bottom > min_protect_gain_price:
                        if self.limit < random_variance_bottom:
                            self.limit = random_variance_bottom
        
        # If we are holding the stock at the end of the timeseries, throw it away.
        # But if ignore_unfinished set to false, execute a sell
        if len(self.buy_dates) != len(self.sell_dates):
            if ignore_unfinished:
                if len(self.buy_dates) > 0:
                    self.buy_dates.pop()
            else:
                self.sell_dates.append(date)
                self.buy_price = None
                        
        return self
                
    def report(self, print_report=True, ignore_lessthanyear=True):
        trades = []
        pct_gains = []
        mpct_gains = []
        
        for i, bd in enumerate(self.buy_dates):
            sd = self.sell_dates[i]
            bp = self.ticker.timeseries[self.ticker.timeseries['Date']==bd]['Close'].values[0]
            sp = self.ticker.timeseries[self.ticker.timeseries['Date']==sd]['Close'].values[0]
            hold = (sd - bd).days
            days = hold
            if ignore_lessthanyear:
                days = 365 if days < 365 else days
            pct = stats.pct(sp, bp, days=days)
            try: 
                mbp = self.market.timeseries[self.market.timeseries['Date']==bd]['Close'].values[0]
                msp = self.market.timeseries[self.market.timeseries['Date']==sd]['Close'].values[0]
                mpct = stats.pct(msp, mbp, days=days)
            except IndexError:
                mbp, msp , mpct = 0, 0, 0
            trades.append((bd, sd, hold, bp, sp, pct, mpct))
            pct_gains.append(pct)
            mpct_gains.append(mpct)
            
        self.trades_list = trades

        if print_report:
            print('Trade report for ${}:'.format(self.ticker.ticker))
            print('{:<10} {:<7} {:<10} {:<7} {:<8} {:<7} {:<7}'.format('Buy', 'Close', 'Sell', 'Close', 'Held', 'Gain%', 'Mkt%'))
            print('{:<10} {:<7} {:<10} {:<7} {:<8} {:<7} {:<7}'.format('----------', '------', '----------', '------', '--------', '-------', '-------'))            
            for bd, sd, hold, bp, sp, pct, mpct in self.trades_list:
                hold_yrs = int(hold/365)
                hold_days = int(hold % 365)
                
                print('{} {:> 7.2f} {} {:> 7.2f} {:>2d}y {:>3d}d {:> 7.2f} {:> 7.2f}'\
                      .format(bd.strftime('%Y-%m-%d'), bp, sd.strftime('%Y-%m-%d'), sp, hold_yrs, hold_days, pct, mpct))
            print('Parameters: min_profit:{}% max_loss:{}% window_long:{}d window_short:{}d'.format(self.min_profit, self.max_loss, self.window_long, self.window_short))
            print('Stats: beta:{:4.2f} std_short:{:4.2f}'.format(self.beta, self.window_short_std))
            print('Average gains: {:>6.2f}%; Market: {:>6.2f}%'.format(np.average(pct_gains), np.average([g for g in mpct_gains if g!=0.0])))
        
        return(self)
    
    def report_plot(self, smoothing=(1, 30), print_report=False):
        if self.trades_list == None:
            self.report(print_report=print_report, ignore_lessthanyear=True)

        sns.set_theme(style="darkgrid")
        sns.set(rc = {'figure.figsize': (15,10)})
        y = stats.rolling_avg(self.ticker.timeseries['Close'], smoothing[0])
        p = sns.lineplot(x="Date", y=y, data=self.ticker.timeseries, alpha=.6)
        p.set_title('Trade report: ${}'.format(self.ticker.ticker))
        p.set_xlim((self.ticker.timeseries['Date'][0], self.ticker.timeseries['Date'][len(self.ticker.timeseries['Date'])-1]))
        p.set_ylim(bottom=0)
        if self.against_market:
            y = stats.rolling_avg(self.ticker.timeseries[self.ticker.column], smoothing[1])
            sns.lineplot(x="Date", y=y, data=self.ticker.timeseries, alpha=.2, ax=p)
            p.legend(labels=[self.ticker.ticker, self.ticker.column])
        else:
            y = stats.rolling_avg(self.market.timeseries['Close'], smoothing[1])
            sns.lineplot(x="Date", y=y, data=self.market.timeseries, alpha=.2, ax=p)
            p.legend(labels=[self.ticker.ticker, self.market.ticker])
        
        for bd, sd, hold, bp, sp, pct, mpct in self.trades_list:
            sns.lineplot(x=[bd, sd], y=[bp, sp], ax=p, color='red')
        return p
    
    def report_table(self, average=False, print_report=False):
        if self.trades_list == None:
            self.report(print_report=print_report, ignore_lessthanyear=True)
            
        columns = ['Ticker', 'Against_Market', 'Min_Profit', 'Max_Loss', \
                   'Window_Short', 'Window_Long', 'Beta', 'Held', 'Pct', 'Mkt_Pct']
        self.trades_df = pd.DataFrame([(self.ticker.ticker, self.against_market, \
            self.min_profit, self.max_loss, self.window_short, self.window_long, \
            self.beta, hold, pct, mpct) for bd, sd, hold, bp, sp, pct, mpct in \
            self.trades_list], columns=columns)
        if average:
            self.trades_df = pd.DataFrame([self.trades_df[columns[2:]].mean()], columns=columns[2:])
            self.trades_df.insert(0, columns[1], self.against_market)
            self.trades_df.insert(0, columns[0], self.ticker.ticker)
        return self.trades_df

##############################################################################
class TradeMarket(object):
    
    def __init__(self, tickers, market, against_market=[False, True], \
                 min_profit=range(1, 11, 9), max_loss=range(25, 51, 25), \
                 window_short=range(1, 16, 14), window_long=range(365, 750, 365)):
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.market = Ticker(market)
        
        self.against_market = against_market
        self.min_profit = min_profit
        self.max_loss = max_loss
        self.window_short = window_short
        self.window_long = window_long
        
        self.report_table = None
        
        self.params = [(t, am, mp, ml, ws, wl) for t in self.tickers for am in against_market \
                  for mp in min_profit for ml in max_loss for ws in window_short \
                  for wl in window_long]
            
    def trade(self, average=False, print_report=False):
        for i, (t, am, mp, ml, ws, wl) in enumerate(self.params):
            logging.info('Running {} of {} trades.'.format(i+1, len(self.params)))
            trade_ticker = TradeTicker(Ticker(t), self.market, min_profit=mp, max_loss=ml, \
                 window_short=ws, window_long=wl, against_market=am)
            self.report_table = pd.concat([self.report_table, trade_ticker.trade() \
                                           .report_table(average=average, print_report=print_report)])
        self.report_table = self.report_table.reset_index(drop=True)
        
        return self
    
    def report_table_save(self, filename='~/Desktop/Server/market_data/TradeMarket/trade_market.csv'):
        self.report_table.to_csv(filename, float_format='%.3f')
        return self
            

##############################################################################
def merge_dfs(tickers, merge_on='Date', how='outer'):
    """
    Takes a list of 2-column (Date-Price) dataframes and merges them based on a
    common column ('Date' by default). The merged dataframe has column names 
    of the merged column, followed by the list of tickers supplied. i.e. 
    ['Date', 'SPY', 'AAPL', 'AMZN'] etc. An earliest date can be set as cutoff 
    point - by default whatever the earliest common date between all columns is
    becomes the cutoff for inner joins. Finally, each column can be converted 
    to percentage values compared to their start prices so we can compare 
    relative gains or losses. If diff_to_first is set to true, the first ticker
    is subtracted from everything else and then dropped. 
    
    So, to get a stock's relative performance to the market, set how=inner,
    first ticker = SPY, pct_convert=True, and diff_to_first as True.

    Parameters
    ----------
    tickers : list of pandas dataframes
        Dataframes of 'Date'-'Close' timeseries to be merged.
    tickers : list of strings
        Ticker names used to label the price columns.
    merge_on : string, optional
        The column to merge on. The default is 'Date'.
    pct_convert : boolean, optional
        Convert each column to pct growth rates. The default is True.
    window_size : int
        The window size over which percentage growth is computed. Default: 0

    Returns
    -------
    merged : pandas dataframe
        The merged dataframe.

    """        
    columns = [merge_on]+[ticker.ticker for ticker in tickers]
    merged = tickers[0].timeseries[['Date', tickers[0].column]]                 # This is our starting ticker
    for ticker in tickers[1:]:              # Porgressively merge all tickers
        merged = merged.merge(ticker.timeseries[['Date', ticker.column]], on=[merge_on], how=how)
    
    merged = merged.sort_values(merge_on)
    merged.columns = columns
    merged = merged.reset_index(drop=True)
    return merged

##############################################################################
def timeseries_df(response, columns):
    """
    Converts a timeries, price tuple list to a pandas dataframe, sorted by timestamp

    Parameters
    ----------
    response : List of (timestamp, price) tuples
        A timeseries list of stock prices.

    Returns
    -------
    df : DataFrame
        DatFrame with Date and Price columns.
    """
    response = sorted(response)
    df = pd.DataFrame(response, columns=columns)
    df['Date'] = pd.to_datetime(df['Date'])
    df[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce').fillna(0)
    df = df.sort_values('Date')            

    return df

##############################################################################
def transform_intraday_df(ticker, column='Close', trim_afterhours=True, \
                          mkt_open=datetime.time(9,30), mkt_close=datetime.time(16,0), \
                          fillna=0):
    """
    Takes an intraday ticker object with 'Date' and any set of columns per timestamp:
    e.g. '2020-01-25 9:30' '3.14' '3.00' '2.95' etc and transforms into a dataframe 
    that retains a single column -- default 'Close' -- but with one row per date
    and a column per timestamp.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    ticker.drop_columns(keep=['Date']+[column])
    if trim_afterhours:
        mask = ticker.timeseries['Date'].apply(lambda d: d.time() < mkt_open or d.time() > mkt_close)
        ticker.timeseries.drop(ticker.timeseries[mask].index, inplace=True)

    rows = []
    row = None
    dates = []
    date = None
    for i, tick in ticker.timeseries.iterrows():
        tick_date = tick['Date'].date()
        tick_time = tick['Date'].time()
        # Starting a new date
        if tick_date != date:
            # If this isn't the first date
            if date != None:
                rows.append(row)
                dates.append(date)
            row = {}
            date = tick_date
        row[tick_time.strftime('%H:%M')] = tick[column]
    df = pd.DataFrame(rows, index=dates).fillna(method='bfill', axis=0)
    df = df.reindex(sorted(df.columns), axis=1)
    return df