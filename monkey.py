import os
# Force matplotlib to not use any Xwindows backend.
if os.name == 'posix':
       import matplotlib
       matplotlib.use('Agg')

from requests import Request, Session
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import date, timedelta
import numpy as np
import pdb
from requests.utils import quote
from io import StringIO

from bokeh.io import output_notebook, show
from bokeh.models import HoverTool
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.plotting import figure, output_file, save
from bokeh.models import Legend

def login():
    url = "https://monkeypython.ch/Account/Login"
    s = Session()
    req = Request('POST', url, data={'UserName': 'ct', 'Password': 'gfhjkm_macos'})
    resp = s.send(req.prepare())
    #    print(resp.status_code)
    return s

def prev_weekday(adate):
    adate -= timedelta(days=1)
    while adate.weekday() > 4: # Mon-Fri are 0-4
        adate -= timedelta(days=1)
    return str(adate.date())


# pivot (col = ) on Month ('M'), Quarter ('Q'), dayofweek ('D-Week'), weekofyear ('W')
# when pivoting specify number of graph columns with col_wrap
# example:
# X = monkey.getFundamentals('JODI,Saudi Arabia,exports,Crude oil')
# Y = monkey.getFundamentals('JODI,Saudi Arabia,runs,Crude oil')
# monkey.plotXY(X, Y)
# monkey.plotXY(X, Y, col= 'Q', col_wrap = 2)
def plotXY(X, Y, x_val = 'Value', y_val= 'Value_y', col = None, col_wrap= None, hue = 'Year', fit_reg= False, palette="Set2"):

    X['Year'] = X.index.year
    Y['Year'] = Y.index.year

    today = dt.datetime.now()

    data = X.join(Y,rsuffix='_y')

    data.loc[data.index == prev_weekday(today), 'Year'] = 'Last'

    data['M'] = data.index.month

    data['Q'] = data.index.quarter

    data['D-Week'] = data.index.dayofweek

    data['Y'] = data.index.year

    data['W'] = data.index.weekofyear

    ax = sns.lmplot(x = x_val, y= y_val, col = col , col_wrap= col_wrap, hue = 'Year', data = data, fit_reg= fit_reg, palette="Set2")
    return ax


def plot(data_frame, param = 'Value', delay_plot = False, loc = 111,  title = 'myPlot'):
       plotFrame(data_frame, param = param, delay_plot = delay_plot, loc = loc,  title = title)

def plotWeb(data_frame, param = 'Value', delay_plot = False, loc = 111,  title = 'myPlot'):
    plotFrame(data_frame, param = param, delay_plot = delay_plot, loc = loc,  title = title)
    img = StringIO.StringIO()
    plt.savefig(img, format='png',bbox_inches='tight')
    img.seek(0)

    return img

def plotFrame(data_frame, param = 'Value', delay_plot = False, loc = 111,  title = 'myPlot', years_bold = []):

    data_frame.index = pd.to_datetime(data_frame.index)

    #if(data_frame.columns.contains('Value') == False and param == 'Value' and data_frame.columns.contains('Volume') == True):
    #    param = 'Volume'

    years_list = data_frame['Year'].unique()
    max_year = years_list.max()
    max_series_start = data_frame[data_frame['Year'] == max_year].index.min()

    fig = plt.figure()
    ax = fig.add_subplot(loc)
    ax.xaxis_date()
    myFmt = mdates.DateFormatter('%b')
    ax.xaxis.set_major_formatter(myFmt)
    ax.grid(b=True, which='major', color='202020', linestyle='--')

    plt.title(title)

    cpool = ['#ff0000', '#0000ff', '#ffa500', '#00ff00', '#a5a2a2',
             '#00ffff', '#ff7f50', '#556f2f', '#ff00ff', '#6495ed',
             '#800080', '#bdb76b', '#808080', '#219774', '#8086d9',
             '#00ff80', '#ffb7ff', '#ff8080', '#219700', '#0086d9']

    x_min_list = []
    x_max_list = []
    years_bold = checkProducts(years_bold)
    if years_bold == []:
        years_bold.append(str(max_year))
    for year in reversed(years_list):
        serie = data_frame[data_frame['Year'] == year]
        serie.sort_index(inplace = True)
        start_date = dt.datetime(max_series_start.year - max_year + year, max_series_start.month, max_series_start.day, 0, 0)

        x = (serie.index - start_date + max_series_start).tolist()
        y = serie[param].tolist()

        x_min_list.append(min(x))
        x_max_list.append(max(x))

        line_width = 1
        aplpha = 0.7
        if str(year) in years_bold:
            line_width = 2.5
            aplpha = 1

        color_index = max_year - year

        ax.plot(x, y, label = str(year), color= cpool[color_index], linewidth = line_width,alpha=aplpha, zorder = year)

    x_min = min(x_min_list) + dt.timedelta(days=-5)
    x_max = max(x_max_list) + dt.timedelta(days=5)

    ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    ax.set_autoscaley_on(False)
    ax.set_xlim([x_min, x_max])

    if delay_plot is False:
        plt.show()

    return ax


#    interactive plot for Jupyter notebook
def bokehPlot(data_frame, param='Value', title=None, plot_height=550, plot_width=800, html = False):
    data_frame.index = pd.to_datetime(data_frame.index)

    if ('Year' not in data_frame.columns):
        data_frame['Year'] = data_frame.index.year

    years_list = data_frame['Year'].unique()
    max_year = years_list.max()
    max_series_start = data_frame[data_frame['Year'] == max_year].index.min()

    p = figure(x_axis_type="datetime", title=title, plot_height=plot_height, plot_width=plot_width, toolbar_location='above')
    p.xaxis.formatter = DatetimeTickFormatter(months=["%b"])
    p.ygrid.grid_line_alpha = 0.8
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Value'

    cpool = ['#ff0000', '#0000ff', '#ffa500', '#00ff00', '#a5a2a2',
             '#00ffff', '#ff7f50', '#556f2f', '#ff00ff', '#6495ed',
             '#800080', '#bdb76b', '#808080', '#219774', '#8086d9',
             '#00ff80', '#ffb7ff', '#ff8080', '#219700', '#0086d9']

    x_min_list = []
    x_max_list = []
    for year in reversed(years_list):
        serie = data_frame[data_frame['Year'] == year]
        serie.sort_index(inplace=True)
        start_date = dt.datetime(max_series_start.year - max_year + year, max_series_start.month, max_series_start.day,
                                 0, 0)

        x = (serie.index - start_date + max_series_start).tolist()
        y = serie[param].tolist()

        x_min_list.append(min(x))
        x_max_list.append(max(x))

        line_width = 1
        alpha = 0.7
        if year == max_year:
            line_width = 2.5
            alpha = 1

        color_index = max_year - year

        p.line(x, y, line_color=cpool[color_index], legend=str(year), line_width = line_width, line_alpha= alpha)

    p.legend.click_policy = "hide"

    new_legend = p.legend[0]
    p.legend[0].plot = None
    p.add_layout(new_legend, 'right')

    p.add_tools(HoverTool(tooltips=[('Date', '@x{%F}'), ('Value', '@y')], formatters={"x": "datetime"}))

    show(p)

    if html == True:
        output_file("plot.html")
        save(p)



#  df = monkey.getSeag(63,1,['Z17','M18'])
#  df1 = monkey.normalizeSeag(df, 1) #
#  monkey.plotFrame(df1)
#  monkey.getHistorical([59],[1],['Q317','Q317'], startDate = '2011/1/1', unit = 0)
def normalizeSeag(df, month, lagyears=1):
    # compute the month of the last observation in the year previous to the current year
    lastyear = np.max(df['Year'])
    lastdatemonth = df[df['Year']==lastyear-1].tail(1).index[0].month

    #if year of expiration and last observation are different have to keep track
    if (df[df['Year']==lastyear-1].tail(1).index[0].year == lastyear-1):
        yeargap = 0
    else:
        yeargap = 1

    if month < lastdatemonth:
        yeardiff = lagyears - 1 + yeargap
    else:
        yeardiff = lagyears + yeargap
    df = df.reset_index()
    df['StartDate'] = df.apply(lambda x: dt.date(int(x['Year'] - yeardiff),month,1), axis=1)
    df = df[pd.to_datetime(df['Date'])>pd.to_datetime(df['StartDate'])]
    df.drop(['StartDate'], axis=1, inplace=True)

    dffirst = df.groupby('Year').first()
    dffirst.rename(columns={'Value': 'FirstValue'}, inplace=True)
    dffirst['Year'] = dffirst.index

    result = pd.merge(df, dffirst, on='Year')
    result['Value'] = result['Value'] - result['FirstValue']
    result.drop(['FirstValue', 'Date_y'], axis=1, inplace=True)

    result.rename(columns={'Date_x': 'Date'}, inplace=True)
    result = result.set_index(['Date'])
    return result

# df = monkey.getFundamentals('DOE,PADD2,runs,Crude Oil')
# df1 = monkey.normalizeFundamentals(df,3) # normalize to March
# monkey.plotFrame(df1)
def normalizeFundamentals(df, month):
    df['Date'] = df.index
    df = df[df['Date'].dt.month >= month]
    dffirst = df.groupby('Year').first()
    dffirst.rename(columns={'Value': 'FirstValue'}, inplace=True)
    dffirst['Year'] = dffirst.index
    result = pd.merge(df, dffirst, on='Year')
    result['Value'] = result['Value'] - result['FirstValue']
    result.rename(columns={'Date_x': 'Date'}, inplace=True)
    result = result.set_index(['Date'])
    return result


# df = monkey.getCurve([53],[1],'2017/09/1')
def getCurve(products, coeffs, asof, unit = 0, session=None):
    if session is None:
        ses = login()
    else:
        ses = session

    products_list = checkProducts(products)

    coeff_list = checkCoeffs(coeffs)

    tenors = []

    asof_string = ""
    if isinstance(asof, dt.date):
        asof_string = asof.strftime("%Y-%m-%d")
    else:
        asof_string = asof

    base_url = "https://monkeypython.ch/api/values/"
    request_url = base_url + \
        'getcurve?products=' + ','.join(products_list) + \
        '&coeffs=' + ','.join(coeff_list) + \
        '&tenors=' + ','.join(tenors)+'&unit=' + str(unit) + \
        '&asof=' + asof_string

    response = ses.get(request_url)

    if session is None:
        ses.close()

    json_response = response.json()

    df = pd.DataFrame(json_response, columns=['Date', 'DaysToExpiry', 'Value', 'Year'])
    #df = pd.DataFrame(json_response, columns=['Date', 'Value', 'Year'])

    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)

    return df


# monkey.getFundamentals('DOE,PADD2,runs,Crude Oil')
def getFundamentals(params, period = 'D', col = 'Value', session=None):

    if session is None:
        ses = login()
    else:
        ses = session

    base_url = "https://monkeypython.ch/api/values/"
    request_url = base_url + 'getfundamentals?fundamentalString=' + quote(params,safe='')
    # request_url.replace(" ", "%20")

    response = ses.get(request_url)
    json_response = response.json()

    if session is None:
        ses.close()
    df = pd.DataFrame()
    if "CUSTOM" not in params and "pivot" not in params:    #this is for all dataframes that have only unique dates and a single value column
        df = pd.DataFrame(json_response, columns=['Date', 'Value', 'Year'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.resample(period).mean()
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        df['Year'] = df.index.year
        df.rename(columns={'Value':col}, inplace =True)
    else:
        df = pd.DataFrame(json_response)
        if(len(params.split(','))==2):#if there are only two components in params, this is a dataframe that doesn't need to have date as index
            return df
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        #df = df.resample(period).mean()
        #df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        df.rename(columns={'Value': col}, inplace=True)
    return df



def getFundamentalsMatrix(params, period = 'D', session=None):

    if session is None:
        ses = login()
    else:
        ses = session

    base_url = "https://monkeypython.ch/api/values/"
    request_url = base_url + 'getfundamentalsMatrix?fundamentalString=' + quote(params,safe='')
    # request_url.replace(" ", "%20")

    response = ses.get(request_url)
    json_response = response.json()

    if session is None:
        ses.close()

    df = pd.DataFrame(json_response)
    df['Date'] = pd.to_datetime(df['Date'])

    df = df.set_index('Date')

    #df = df.resample(period).mean()
    #df.dropna(inplace=True)
    df['Year'] = df.index.year
    df.index = pd.to_datetime(df.index)

    return df

def getHistorical(products, coeffs, tenors, unit = 0, startDate = None, session = None):

    if session is None:
        ses = login()
    else:
        ses = session

    if startDate is None:
        today = dt.datetime.today()
        startDate = dt.datetime(today.year - 6, today.month, today.day)

    products_list = checkProducts(products)
    coeff_list = checkCoeffs(coeffs)
    tenors_list = checkTenors(tenors)

    start_date = pd.to_datetime(startDate)

    #'gethistorical?products=51&coeffs=1&tenors=M%2B7,M%2B10&unit=0&startDate=2000-01-01'
    base_url = "https://monkeypython.ch/api/values/"
    request_url = base_url + \
        'gethistorical?products=' + ','.join(products_list) + \
        '&coeffs=' + ','.join(coeff_list) + \
        '&tenors=' + ','.join(tenors_list)+'&unit=' + str(unit) + \
        '&startDate=' + str(start_date.year) + '-' + str(start_date.month) + '-' + str(start_date.day)

    response = ses.get(request_url)
    json_response = response.json()

    #print(request_url)
    #print(json_response)

    if session is None:
        ses.close()

    df = pd.DataFrame(json_response, columns=['Date', 'Value', 'Year'])

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.set_index('Date')

    df['Year'] = df.index.year

    return df


#  df = monkey.getSeag([53],[1],['M18', 'Z18'])
#  df = monkey.getSeag([53],[1],['M+1','M+11'])
def getSeag(products, coeffs, tenors, unit=0, window=1, lookbackcontracts=16, session=None,daysToExpiry=False):

    if session is None:
        ses = login()
    else:
        ses = session

    products_list = checkProducts(products)
    coeff_list = checkCoeffs(coeffs)
    tenors_list = checkTenors(tenors)

    is_rolling = False
    if tenors[0].find('%2B') == 1:
        is_rolling = True

    base_url = "https://monkeypython.ch/api/values/"
    request_url = base_url + \
        'getseasonal?products=' + ','.join(products_list) + \
        '&coeffs=' + ','.join(coeff_list) + \
        '&tenors=' + ','.join(tenors_list)+'&unit=' + str(unit) + \
        '&yearsofhistory=' + str(window) + '&lookbackcontracts=' +  \
        str(lookbackcontracts)

    response = ses.get(request_url)
    json_response = response.json()

    if session is None:
        ses.close()

    df = None
    if(daysToExpiry):
        df = pd.DataFrame(json_response, columns=['Date', 'Value', 'Year','DaysToExpiry'])
    else:
        df = pd.DataFrame(json_response, columns=['Date', 'Value', 'Year'])

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.set_index('Date')

    if is_rolling == True:
        df['Year'] = df.index.year

    return df


def getMenuItems(session=None):
     
    if session is None:
        ses = login()
    else:
        ses = session
  
    base_url = "https://monkeypython.ch/api/values/"
    request_url = base_url + 'getmenuitems'

    response = ses.get(request_url)
    json_response = response.json()
    

    if session is None:
        ses.close()
        
    df = pd.DataFrame(json_response)
    
    return df


# df = monkey.getSeagVolume([54],['M18'])
def getSeagVolume(products, tenors, window=1, lookbackcontracts=12, session=None):

    if session is None:
        ses = login()
    else:
        ses = session

    products_list = checkProducts(products)
    tenors_list = checkTenors(tenors)

    base_url = "https://monkeypython.ch/api/values/"
    request_url = base_url + \
        'getseasonalvolume?products=' + ','.join(products_list) + \
        '&tenors=' + ','.join(tenors_list) + \
        '&yearsofhistory=' + str(window) + '&lookbackcontracts=' +  \
        str(lookbackcontracts)

    response = ses.get(request_url)
    json_response = response.json()

    if session is None:
        ses.close()

    df = pd.DataFrame(json_response, columns=['Date', 'Volume', 'OpenInterest', 'Year', 'DaysToExpiry'])

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.set_index('Date')

    return df


def getMatrixPrices(products, tenors, unit=0, window=2,lookbackcontracts=12, exceptions = None, session=None):

    if session is None:
        ses = login()
    else:
        ses = session
    
    products_list = checkProducts(products)
    tenors_list = checkTenors(tenors)

    base_url = "https://monkeypython.ch/api/values/"
    request_url = base_url + \
        'getmatrixprices?products=' + ','.join(products_list) + \
        '&tenors=' + ','.join(tenors_list)+'&unit=' + str(unit) + \
        '&yearsofhistory=' + str(window) + '&lookbackcontracts=' +  \
        str(lookbackcontracts)
    
    if exceptions is not None:
        request_url = request_url + "&exceptions="+ ",".join(exceptions)
        
    response = ses.get(request_url)
    json_response = response.json()

    if session is None:
        ses.close()

    df = pd.DataFrame(json_response)

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.set_index('Date')

    return df


def getMatrixSingles(products, tenors, unit=0, window=2,lookbackcontracts=12, exceptions=None, session=None):

    if session is None:
        ses = login()
    else:
        ses = session

    products_list = checkProducts(products)
    tenors_list = checkTenors(tenors)

    base_url = "https://monkeypython.ch/api/values/"
    request_url = base_url + \
        'getmatrixsingles?products=' + ','.join(products_list) + \
        '&tenors=' + ','.join(tenors_list)+'&unit=' + str(unit) + \
        '&yearsofhistory=' + str(window) + '&lookbackcontracts=' +  \
        str(lookbackcontracts)
    
    if exceptions is not None:
        request_url = request_url + "&exceptions="+ ",".join(exceptions)
        
    response = ses.get(request_url)
    json_response = response.json()

    if session is None:
        ses.close()

    df = pd.DataFrame(json_response)

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.set_index('Date')

    return df

def getMatrix(products, coeffs, tenors, unit=0, window=2,lookbackcontracts=12, session=None):

    if session is None:
        ses = login()
    else:
        ses = session

    products_list = checkProducts(products)
    coeff_list = checkCoeffs(coeffs)
    tenors_list = checkTenors(tenors)

    base_url = "https://monkeypython.ch/api/values/"
    request_url = base_url + \
        'getmatrix?products=' + ','.join(products_list) + \
        '&coeffs=' + ','.join(coeff_list) + \
        '&tenors=' + ','.join(tenors_list)+'&unit=' + str(unit) + \
        '&yearsofhistory=' + str(window) + '&lookbackcontracts=' +  \
        str(lookbackcontracts)

    response = ses.get(request_url)
    json_response = response.json()

    if session is None:
        ses.close()

    df = pd.DataFrame(json_response, columns=['Date', 'Value', 'Year'])

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.set_index('Date')

    return df

def getMatrixHistorical2(products, coeffs, tenors, start_date, end_date, unit=0, session=None):

    if session is None:
        ses = login()
    else:
        ses = session

    products_list = checkProducts(products)
    coeff_list = checkCoeffs(coeffs)
    tenors_list = checkTenors(tenors)
    
    if isinstance(start_date, dt.date):
        start_date = start_date.strftime("%Y/%m/%d")

    if isinstance(end_date, dt.date):
        end_date = end_date.strftime("%Y/%m/%d")


    base_url = "https://monkeypython.ch/api/values/"
    request_url = base_url + \
        'getmatrixhistorical2?products=' + ','.join(products_list) + \
        '&coeffs=' + ','.join(coeff_list) + \
        '&tenors=' + ','.join(tenors_list)+'&unit=' + str(unit) + \
        '&startDate='+start_date+'&endDate='+end_date

    response = ses.get(request_url)
    json_response = response.json()

    if session is None:
        ses.close()

    df = pd.DataFrame(json_response, columns=['Date', 'Value', 'Year'])

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.set_index('Date')

    return df

def getMatrixSpreads(products, tenors, unit=0, window=2,lookbackcontracts=12, session=None,coeffs=None):

    if session is None:
        ses = login()
    else:
        ses = session

    products_list = checkProducts(products)
    coeff_list = checkCoeffs(coeffs)
    tenors_list = checkTenors(tenors)

    base_url = "https://monkeypython.ch/api/values/"
    request_url = base_url + \
        'getmatrixspreads?products=' + ','.join(products_list)
    if coeffs is not None:
        request_url=request_url+'&coeffs=' + ','.join(coeff_list)
        
    request_url=request_url+'&tenors=' + ','.join(tenors_list)+'&unit=' + str(unit) + \
        '&yearsofhistory=' + str(window) + '&lookbackcontracts=' +  \
        str(lookbackcontracts)

    response = ses.get(request_url)
    json_response = response.json()

    if session is None:
        ses.close()

    df = pd.DataFrame(json_response)

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.set_index('Date')

    return df
	
def CompareYears(in_df, threshold = 0.2):

       max_year = in_df.Year.max()

       frame_v = _getValuesFrame(in_df)
       frame_d = _getDinamicsFrame(in_df)

       result = pd.merge(frame_v, frame_d, on = 'Year')
       result['Sum'] = result['Diff'] + result['Value_y']
       result.Sum = result.Sum/result.Sum.max()

       similar_years = result[result['Sum'] <= threshold].Year.values
       if len(similar_years) > 0:
              similar_years = np.delete(similar_years, 0)
       unlike_years = result[result['Sum'] > threshold].Year.values

       print('Similar years: ', similar_years)
       print('Unlike years: ', unlike_years)

       return similar_years, unlike_years


def _getDinamicsFrame(in_df):
       max_year = in_df.Year.max()
       min_year = in_df.Year.min()

       date2 = pd.to_datetime('today')
       date1 = date2  - pd.DateOffset(months = 3)

       df1 = in_df[in_df.Year == max_year]

       last_frame = _prepareData(df1, date1, date2)
       last_frame['Value'] = last_frame['Value'] - last_frame.Value[0:10].mean()

       frame = pd.DataFrame(columns = ['Year','Value'])

       for i in range(0, max_year - min_year + 1):
              year = max_year - i

              df1 = in_df[in_df.Year == year]
              df1.index = df1.index + pd.DateOffset(years = i)

              current_frame = _prepareData(df1, date1, date2)
              current_frame['Value'] = current_frame['Value'] - current_frame.Value[0:10].mean()

              d = (last_frame.Value - current_frame.Value)
              d = d*d

              frame.loc[i] = [year, np.sqrt(d.values).sum()]


       frame.Value = frame.Value/ frame.Value.max()

       return frame


def _getValuesFrame(in_df, in_number_of_points = 40):
       max_year = in_df.Year.max()
       min_year = in_df.Year.min()

       today = pd.to_datetime('today')

       frame = pd.DataFrame(columns = ['Year','Value','Diff'])
       current_value = _getValues(in_df, max_year, today, in_number_of_points)

       for i in range(0, max_year - min_year + 1):
              year = max_year - i
              date = today - pd.DateOffset(years = i)

              value = _getValues(in_df, year, date, in_number_of_points)
              frame.loc[i] = [year, value, np.abs(current_value - value)]

       index = np.abs(frame.Diff.values)<np.abs(frame.Diff.values.std())
       similar_years = frame[index].Year.values
       unlike_years = frame[~index].Year.values

       frame.Diff = frame.Diff/ frame.Diff.max()

       return frame


def _getValues(in_df, in_year, in_date, in_number_of_points):
       frame = in_df[in_df.Year == in_year]
       vector = frame.index <= pd.to_datetime(in_date)
       values = frame.Value[vector].values

       return values[-in_number_of_points:].mean()

def _prepareData(in_df, in_date1, in_date2, in_ma_window = 5):

       df = in_df[~in_df.index.duplicated(keep='first')]
       df = df[(df.index > in_date1 -  pd.DateOffset(days = 10)) & \
               (df.index < in_date2 + pd.DateOffset(days = 10))]
       frame = pd.DataFrame(df.resample('D').interpolate(method = 'time'))
       frame['Value'] = frame['Value'].rolling(window = in_ma_window).mean()

       frame = frame[(frame.index >= in_date1) & (frame.index < in_date2)]

       #frame['X'] = np.array(frame.reset_index().index, dtype='float')

       return frame


def checkTenors(intenors):

    if type(intenors) is not list:
        tenors = []
        tenors.append(intenors)
    else:
        tenors = intenors

    if tenors[0].find('+') == 1:
        tenors = [tenor.replace('+', '%2B') for tenor in tenors]

    return tenors


def checkProducts(inproducts):

    if type(inproducts) is not list:
        products = []
        products.append(inproducts)
    else:
        products = inproducts

    products_list = []
    for product in products:
        if isinstance(product, str) is False:
            products_list.append(str(product))
        else:
            products_list.append(product)

    return products_list


def checkCoeffs(incoeffs):

    if type(incoeffs) is not list:
        coeffs = []
        coeffs.append(incoeffs)
    else:
        coeffs = incoeffs

    coeff_list = []
    for coeff in coeffs:
        if isinstance(coeff, str) is False:
            coeff_list.append(str(coeff))
        else:
            coeff_list.append(coeff)

    return coeff_list