from requests import Request, Session
import pandas as pd
import json
from requests.utils import quote
import datetime
import os, errno

#url ='http://localhost:62757/'
url = "https://monkeypython.ch/"
    
def login():        
    login_url = url+'Account/Login'
    s = Session()
    req = Request('POST', login_url, data={'UserName':'ct','Password':'gfhjkm_macos'})   
    resp = s.send(req.prepare())
#    print(resp.status_code)
    return s

def getSeriesDesc(ses=None):
    if(ses==None):ses=login()
    req = url+'api/values/getcustomseriesdescription'
    response=ses.get(req) #get Custom series description
    #print (response.status_code)
    custom_desc = pd.DataFrame(response.json())
    return custom_desc

def getArray(fundamentalString,date=None,ses=None):
    if(ses is None):ses=login()
    safeString = convertToSafeString(fundamentalString)
    if(date is None): date = '2017-09-28'
    req=url+'api/values/getcustomseries?fundamentalString='+ safeString+'&date='+date
    jsonResponse = ses.get(req)    
    #print jsonResponse.json()
    return pd.DataFrame(jsonResponse.json())
    #return pd.read_json(jsonResponse.json(),orient='table')

def getDataFrame(name, date = None, ses=None):
    if isinstance(name, int ): return getDataFrame_by_SeriesID(name,date,ses)
    if isinstance(name, str ): return getDataFrame_by_Name(name,date,ses)
    return
    
def getDataFrame_by_Name(fundamentalString,date=None,ses=None):
    if(ses is None):ses=login()
    safeString = convertToSafeString(fundamentalString)  
    req = ''
    if(date is None): 
        req=url+'api/values/getcustomseries?fundamentalString='+ safeString
    else:
        req=url+'api/values/getcustomseries?fundamentalString='+ safeString+'&date='+convertDateToStr(date)
    jsonResponse = ses.get(req)
    if(jsonResponse.status_code==200):
       #return pd.read_json(jsonResponse.content,orient='table')
       return pd.DataFrame(jsonResponse.json()['data'])
    else:
        print('The requested series does not exist')
        return "The requested series doesn't exist"

def getDataFrame_by_SeriesID(seriesID,ses=None):  
    if(ses is None):ses=login()
    req=url+'api/values/getcustomseries?seriesID='+ str(seriesID)    
    jsonResponse = ses.get(req)
    if(jsonResponse.status_code==200):
       return pd.DataFrame(jsonResponse.json()['data'])
       #return pd.read_json(jsonResponse.content,orient='table')
    else:
        print('The requested series does not exist')
        return "The requested series doesn't exist"

def saveDataFrame(name,series,date=None,ses=None,comment=None):    
    resp = postData_withDate(name=name,sequence=series,date=date,dataType='Panda Dataframe',ses=ses,comment=comment)
    if(resp.status_code==201):print('Successfully posted the series')
    else:print("Something went wrong while posting the series")
    return resp

def saveScript(name,file,date=None,ses=None,comment=None):
    with open(file,'r') as myfile:
        series=myfile.read()
    #print (series)    
    resp = postData_withDate(name=name,sequence=series,date=date,dataType='Python Script',ses=ses,comment=comment)    
    if(resp.status_code==201):print('Successfully saved the file')
    else:print("Something went wrong while saving the file")
    return resp

def getScript(name,path,date=None,ses=None):
    if(ses is None):ses=login()
    safeString = convertToSafeString(name)  
    req = ''
    if(date is None): 
        req=url+'api/values/getcustomseries?fundamentalString='+ safeString
    else:
        if isinstance(date, int ): 
            desc = getSeriesDesc()
            filtered = (desc[(desc['DataType']=='Python Script')&(desc['FundamentalString']==name)]).sort_values(['Date'],ascending=True)
            numberOfRows = len(filtered.index)
            if(numberOfRows>(abs(date))):
                seriesID = (filtered.iloc[abs(date)])['SeriesID']
                req=url+'api/values/getcustomseries?seriesID='+ str(seriesID)
            else:
                print("There are no so many versions as requested. Number of available versions is "+str(numberOfRows))
                return
        else:
            req=url+'api/values/getcustomseries?fundamentalString='+ safeString+'&date='+convertDateToStr(date)
    response = ses.get(req)
    if(response.status_code==200):
        silentremove(path)
        with open(path, "ab") as text_file:
            text_file.write(response.content)
        print("The requested file saved in "+path)
        return "The requested file saved in "+path
    else:
        print('The requested file does not exist. Requested:'+name)
        return "The requested file doesn't exist. Requested:"+name
 
def postData_withDate(name,sequence,dataType,date=None,ses=None,comment=None): 
    if(ses is None):ses=login()
    safeString = convertToSafeString(name)
    safeDataType = convertToSafeString(dataType)
    date = convertDateToStr(date)  
    if(date is None):date=getNowDate()
    req = url+'api/values/postcustomseries?fundamentalString=' + safeString+'&date='+date+'&dataType='+safeDataType
    if(comment is not None):
        req = req+'&comment='+convertToSafeString(comment)
    if(dataType!='Python Script'):
        postJson=convertJSON(sequence)   
    else: 
        print("Python Script identified")
        postJson = sequence        
    resp = ses.post(req,data=postJson)  
    #print(resp.status_code)
    return resp

def delete(seriesID,ses=None):
    if(ses==None):ses=login()
    req = url+'api/values/deletecustomseries?SeriesID='+ str(seriesID)
    resp = ses.delete(req)  
    if(resp.status_code==200):print('Successfully deleted the series')
    else:print("Something went wrong while deleting the series")
    return resp

def updateName(new_name,seriesID,ses=None):
    if(ses is None):ses=login()
    req = url+'api/values/updatecustomseries?fundamentalString='+ new_name+'&seriesID='+str(seriesID)
    resp = ses.post(req)  
    if(resp.status_code==200):print("The name was successfully changed")
    else:print("Something went wrong while updating the series name")
    return resp

def convertToSafeString(s):
    return quote(s,safe='')

def getSeriesID_by_Name(name,ses=None):
    if(ses is None):ses=login()
    series = getSeriesDesc(ses)
    numberOfRows = len(series[series.FundamentalString == name].index)
    seriesID=-1
    if(numberOfRows>0):
        seriesID = series[series.FundamentalString == name].iloc[0].SeriesID
        #print (seriesID)        
    return seriesID

def convertJSON(incoming_data):
    postJson = 'empty'
    if isinstance(incoming_data,pd.DataFrame):
        print('Panda dataframe identified')
        postJson=incoming_data.to_json(orient='table')
        #print postJson
    else:
        print('Array identified')
        postJson = json.dumps(incoming_data)
    return postJson
    
def getNowDate():
    now = convertDateToStr(datetime.datetime.now())
    return now

def convertDateToStr(date):
    if isinstance(date,datetime.datetime):
        dt = datetime.datetime(date.year,date.month,date.day)
        dt = datetime.datetime.strftime(dt,"%Y-%m-%d")
        #print(dt)
        return dt
    else: return date



def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred