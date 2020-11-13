#! /usr/bin/env python
# coding:utf-8
'''
Created on 2020年1月2日

ISO 8601 解释器——转换对象到文本
@author: Lichengang
'''

import re
import datetime
from timeexp.postprocess import DATE, TIME, RANGE, DURATION
from timeexp.datetime_extra import *
from calendar import monthrange

# 定义表达式解析的常量
DOCUMENT_YEAR = "DDDD"
LAST_YEAR = "LLLL"
DOCUMENT_HOUR = DOCUMENT_DAY = DOCUMENT_MONTH = "DD"
LAST_HOUR = LAST_DAY = LAST_MONTH = "LL"
DOCUMENT_WEEKDAY = "D"
LAST_WEEKDAY = "L"

RELATIVE = 'RELATIVE'

def parse_year(value, document_time : date, last_time : date):
    ''' 解析年份
    :return int 年份
    '''
    if value == DOCUMENT_YEAR:
        value = str(document_time.year)
    elif value == LAST_YEAR:
        value = str(last_time.year)
    return int(value)

def parse_month(value, document_time : date, last_time : date):
    ''' 解析月份
    :return int 月份
    '''
    if value == DOCUMENT_MONTH:
        value = str(document_time.month)
    elif value == LAST_MONTH:
        value = str(last_time.month)
    return int(value)

def parse_day(value, document_time : date, last_time : date, year = 0, month = 0):
    ''' 获取日期，MX表示月中最后一天
    :return int 日期
    '''
    if value == DOCUMENT_DAY:
        value = str(document_time.day())
    elif value == LAST_DAY:
        value = str(last_time.day())
    elif value == "MX":
        if year == 0:
            year = document_time.year
        if month == 0:
            month = document_time.month
        value = monthrange(year, month)[1]
    return int(value)

def parse_week_num(value, document_time : date, last_time : date):
    ''' 解析星期数
    :return int 星期数
    '''
    if value == DOCUMENT_MONTH:
        value = str(week_num(document_time))
    elif value == LAST_MONTH:
        value = str(week_num(last_time))
    return int(value)

def parse_weekday(value, document_time : date, last_time : date):
    ''' 解析星期数
    '''
    if value == DOCUMENT_WEEKDAY:
        value = str(document_time.isoweekday())
    elif value == LAST_WEEKDAY:
        value = str(last_time.isoweekday())
    return int(value)

def parse_hour(value, document_time : date, last_time : date):
    ''' 解析小时
    :return int 小时
    '''
    if value == DOCUMENT_HOUR:
        value = str(document_time.hour)
    elif value == LAST_HOUR:
        value = str(last_time.hour)
    return int(value)

def parse_minute(value, document_time : date, last_time : date):
    ''' 解析分钟
    :return int 分钟
    '''
    if value == DOCUMENT_MONTH:
        value = str(document_time.minute)
    elif value == LAST_MONTH:
        value = str(last_time.minute)
    return int(value)

def parse_second(value, document_time : date, last_time : date):
    ''' 解析秒
    :return int 秒
    '''
    if value == DOCUMENT_MONTH:
        value = str(document_time.second)
    elif value == LAST_MONTH:
        value = str(last_time.second)
    return int(value)

def parse_date(document_time : date, last_time : date, expression : str):
    ''' 解析expression为日期对象
    :return datetime.date 类型的日期
    '''
    year = ""
    base_date = document_time
    # 年-周序号-周内天数
    match = re.match("(\\w\\w\\w\\w)-W(\\w\\w)-(\\d)", expression, re.IGNORECASE)
    if match:
        year = parse_year(match.group(1), document_time, last_time)
        week_num = parse_week_num(match.group(2), document_time, last_time)
        base_date = weekday_to_ymd(year, week_num, int(match.group(3)))
    else:
        # 年-月-日
        match = re.match("(\\w\\w\\w\\w)-(\\w\\w)-(\\w\\w)", expression, re.IGNORECASE)
        if not match:
            return None
        year = parse_year(match.group(1), document_time, last_time)
        month = parse_month(match.group(2), document_time, last_time)
        day = parse_day(match.group(3), document_time, last_time)
        base_date = date(year, month, day)
    return base_date

def parse_time_point(document_time : date, last_time : date, expression : str):
    ''' 解析时间点信息
    :return datetime 计算出的时间
    '''
    pattern = r"(\w+-\w\w-\w\w)(T\w\w\:\w\w\:\w\w)?"
    match = re.match(pattern, expression)
    if not match:
        return None
    date = parse_date(document_time, last_time, match.group(1))
    hour = minute = second = 0
    if match.lastindex > 1 and match.group(2) != "":
        timematch = re.match(r"T(\d\d)\:(\d\d)\:(\d\d)", match.group(2))
        hour = parse_hour(timematch.group(1), document_time, last_time);
        minute = parse_minute(timematch.group(2), document_time, last_time);
        second = parse_second(timematch.group(3), document_time, last_time);
    base_time = datetime(date.year, date.month, date.day, 
                      hour, minute, second)
    return base_time

def build_duration(length, unit : str):
    if unit == "Y":
        return Duration(years=int(length))
    elif unit == "M":
        return Duration(months=int(length))
    elif unit == "D":
        return Duration(days=int(length))
    elif unit == "H":
        return Duration(hours=int(length))
    elif unit == "N":
        return Duration(minutes=int(length))
    elif unit == "S":
        return Duration(seconds=int(length))

def parse_duration(document_time : date, last_time : date, expression):
    ''' 解析时长信息
    :return Duration 计算出的时间
    '''
    if expression[0] != 'P':
        return None
    pattern = "(\\d+)([YMWDNHS])"
    duration = None
    for match in re.finditer(pattern, expression, re.IGNORECASE):
        new_duration = build_duration(match.group(1), match.group(2))
        duration = new_duration if not duration else duration + new_duration
    return duration

def parse_range(document_time : date, last_time : date, expression):
    ''' 解析时间段信息
        YYYY-MM-DDTHH:mm:ss/YYYY-MM-DDTHH:mm:ss/PxS
    :return dict 计算出的时间
    '''
    range_group = expression.split("/")
    if len(range_group) < 2:
        return None
    begin = end = None
    if range_group[0] != "":
        begin = parse_time_point(document_time, last_time, range_group[0])
    if len(range_group) > 1 and range_group[1] != "":
        end = parse_time_point(document_time, last_time, range_group[1])
    if range_group[2] != "":
        duration = parse_duration(document_time, last_time, range_group[2])
    if not begin:
        begin = duration - end
    if not end:
        end = duration + begin
    if not duration:
        delta = end - begin
        duration = Duration(delta.total_seconds())
    return {"begin" : begin, "end" : end, "duration" : duration}

def parse_relative_point(document_time : date, last_time : date, relative_expression : str):
    ''' 解析相对时间
        YYYY-MM-DDTHH:mm:ss R+xS
    :return datetime.datetime 计算出的时间
    '''
    pattern = r"(\w+-\w+-\w+)(T\d\d:\d\d:\d\d)? R(+|-)(\d+)(\w+)"
    match = re.match(pattern, relative_expression)
    if not match:
        return None
    date = parse_time_point(document_time, last_time, match.group(1))
    duration = build_duration(match.group(4), match.group(5))
    if match.group(3) == '-':
        result_date = duration - date
    else:
        result_date = duration + date
    return result_date

def parse(results : list, document_time : date=None):
    ''' 对所有时间表达进行解析
    '''
    if not document_time:
        document_time = date.today()
    methods = {DATE : parse_date, TIME : parse_time_point, 
               RELATIVE : parse_relative_point,
               RANGE : parse_range, DURATION : parse_duration}
    last_time = document_time
    for result in results:
        method = methods[result['type']]
        value = method(document_time, last_time, result['result'])
        result['value'] = value
        if result['type'] == RELATIVE:
            result['type'] = TIME
        if result['type'] == TIME and value is not None:
            last_time = value.date()
        elif result['type'] == DATE and value is not None:
            last_time = value
    return results

if __name__ == '__main__':
    document_time = date.today()
    last_time = date(2019,1,25)
    print(parse_duration(document_time, last_time, "P03H"))
    print(parse_date(document_time, last_time, "DDDD-W51-01"))
    print(parse_date(document_time, last_time, "DDDD-10-MX"))
    print(parse_date(document_time, last_time, "LLLL-10-25"))
    print(parse_time_point(document_time, last_time, "LLLL-10-25T15:30:00"))
    print(parse_range(document_time, last_time, "LLLL-10-25T15:30:00//P02H30N"))