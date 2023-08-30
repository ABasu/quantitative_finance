#!/usr/bin/env python# encoding=utf8"""A set of functions to make API calls to Alphavantage. https://www.alphavantage.co/documentation/Examples:av.call_api('TSLA', 'TIME_SERIES_DAILY', api_key, parameters={})av.call_api('TSLA', 'TIME_SERIES_INTRADAY', api_key, parameters= \            {'pause_between_queries':5, 'interval':'60min', \             'month_start':'2020_12', 'month_end':'2021_02'})"""import logging, re, requests, timeimport pandas as pdlogging.basicConfig(format = '%(asctime)-25s %(message)s', level = logging.INFO)##############################################################################def call_api(symbol, function, apikey, parameters={}, normalize_column_headings=True):    """    Calls the Alphavantage API. Retrieves JSON data and converts it to a Pandas     DataFrame. Most calls result in a single API query, but INTRADAY data requires    one query per month (unless month_start isn't suuplied in parameters, then     just the most recent data is retrieved').        Parameters    ----------    symbol : string        A ticker for a stock.    function : string, optional        The API function to be called. Look up on alphavantage.            TIME_SERIES_INTRADAY: If month_start, month_end, and pause_between_queries is                 supplied, then results in multiple calls. Otherwise, single call to get the                 latest data.    apikey : string        The API key for alphavantage.            parameters: dict        Parameters to be passed on to the alphavantage API or used inside the api call:             For TIME_SERIES_INTRADAY pause_between_queries is required because it requires             multiple calls, also requires month_start, month_endvalues in YYYY_MM format)        Look up https://www.alphavantage.co/documentation/ for parameters.    normalize_columns_headings: bool        Standardizes column headings in the Pandas dataframe: e.g. 1. open -> Open            Returns    -------    Pandas DataFrame            """    # Requires multiple calls. This function needs the parameters to have at least    # the following values: pause_between_queries, month_start, month_end    if function=='TIME_SERIES_INTRADAY' and ('month_start' in parameters) \        and ('month_end' in parameters) and ('pause_between_queries' in parameters) :        monthly_dataframes = []        pause_between_queries = parameters.pop('pause_between_queries')        month_start = parameters.pop('month_start')        month_end = parameters.pop('month_end')        sy, sm = month_start.split('_')        sy, sm = int(sy), int(sm)        ey, em = month_end.split('_')        ey, em = int(ey), int(em)        # For each month, we issue a call and then compile all responses together        for y in range(sy, ey+1):            if y == sy:                lsm = sm            else:                lsm = 1            if y == ey:                lem = em            else:                lem = 12            if y > sy and y < ey:                lsm = 1                lem = 12            for m in range(lsm, lem+1):                parameters['month'] = '{:04d}-{:02d}'.format(y, m)                url = generate_api_url(function=function, symbol=symbol, parameters=parameters, apikey=apikey)                logging.info('{}: Downloading {}: {:04d}-{:02d}'.format(symbol, function, y, m))                with requests.Session() as s:                    response = s.get(url).json()                    month_df = parse_response(response)                    # If we got a valid dataframe back, save it                    if isinstance(month_df, pd.DataFrame):                        monthly_dataframes.append(month_df)                logging.info('Pausing for {} secs.'.format(pause_between_queries))                time.sleep(pause_between_queries)        dataframe = pd.concat(monthly_dataframes)    else:        url = generate_api_url(function=function, symbol=symbol, parameters=parameters, apikey=apikey)        try:            logging.info('{}: Downloading {}'.format(symbol, function))            response = requests.get(url).json()            dataframe = parse_response(response)        except Exception as e:            logging.exception(e)            response = None    if normalize_column_headings:        dataframe.columns = [re.sub('^\d\. ', '', c).title() for c in dataframe.columns]            return dataframe ##############################################################################def generate_api_url(symbol, apikey, function, parameters={}):    """    Generates an Alphavantage API url.        Parameters    ----------    symbol : string        A ticker for a stock.    function : string, optional        The API function to be called. Look up on alphavantage.    apikey : string, optional        The API key for alphavantage. The default is apikey.    Returns    -------    url : string        The formatted url for the API call.    """    parameters = ['{}={}'.format(p, v) for p, v in parameters.items()]    parameters = '&'+'&'.join(parameters)    url = 'https://www.alphavantage.co/query?function={}&symbol={}&apikey={}{}'.format(function, symbol, apikey, parameters)    return url    ##############################################################################def get_api_key(config_file):    """    Reads the api key from a config file.        Parameters    ----------    config_file : str        A file containing the value:key pair 'apikey=APIKEY'. The default is 'apikey.cfg'.    Returns    -------    String apikey.    """    apikey = [l for l in open(config_file, 'r').readlines() if re.search('^apikey\s*=', l)][0].split('=')[1].strip().strip("'")        return apikey##############################################################################def parse_response(response):    """    Depending on the function call, returns either a timeseries as a list of     tuples or a dictionary (for OVERVIEW).        Parameters    ----------    response : dict        A dictionary of the API response.    function : TYPE, optional        The API function to be called. Look up on alphavantage. The default is 'TIME_SERIES_WEEKLY_ADJUSTED'.    Returns    -------    List of tuples. Or dictionary        Returns (date, price) list of Tuples. Or dictionary with over view         key, value pairs.    """    # To be a valid response we need two keys and one of them would be "Meta Data"    if (len(response.keys()) == 2) and ('Meta Data' in response):        metadata = response.pop('Meta Data')        dataframe = pd.DataFrame.from_dict(response[list(response.keys())[0]], orient='index').sort_index()        for md in metadata:            dataframe.attrs[md] = metadata[md]    else:        logging.error('{}'.format(response))        return None        dataframe.index = pd.to_datetime(dataframe.index)    dataframe = dataframe.astype(float)        return dataframe