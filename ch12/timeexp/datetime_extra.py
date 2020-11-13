#! /usr/bin/env python
# coding:utf-8
'''
Created on 2020年1月9日

@author: Lichengang
'''

from datetime import datetime, date, timedelta

    
def _week_param(year):
    '''
    · 计算“星期日历表示法”的关键参数
    Returns:
        tuple    当年1月1日的序数, 当年第一周开始时间与1月1日相差多少
    '''
    first_day = date(year, 1, 1)
    # 如果1月1日是星期一～星期四，它所在的星期就是第一周
    sign_weekday = first_day.weekday()
    start_offset = -sign_weekday
    if start_offset < -3:
        start_offset += 7
    return first_day.toordinal(), start_offset
    
def week_num(date=date.today()):
    ''' 提取本日所属的周 '''
    base_ordinal, start_offset = _week_param(date.year)
    return (date.toordinal() - base_ordinal - start_offset) // 7
    
def weekday_to_ymd(year, week_no, iso_weekday):
    ''' 星期日历表示法转换为年月日 '''
    base_ordinal, start_offset = _week_param(int(year))
    ordinal = base_ordinal + int(week_no-1)*7 + iso_weekday - 1 + start_offset
    return date.fromordinal(ordinal)


class Duration(object):
    '''
    $ 在timedelta相比增加了年和月的计算 
    '''

    def __init__(self, years=0, months=0, days=0, 
                 seconds=0, minutes=0, hours=0, weeks=0):
        '''
        Constructor
        '''
        self._years = years
        self._months = months
        days += weeks*7
        extradays = hours // 24
        days += extradays
        hours = hours - extradays*24
        seconds += minutes*60 + hours*3600
        self._days = days
        self._seconds = seconds

    # Read-only field
    @property
    def years(self):
        """years"""
        return self._years

    @property
    def months(self):
        """months"""
        return self._months

    @property
    def days(self):
        """days"""
        return self._days

    @property
    def seconds(self):
        """seconds"""
        return self._seconds
        
    def normalize_month(self, year, month):
        if month < 0 or month > 12:
            yeardelta, month = divmod(month, 12)
            year += yeardelta
        elif month == 0:
            return (-1, 12)
        return year, month
    
    def __add__(self, other):
        if isinstance(other, timedelta):
            return Duration(years=self._years, months=self._months,
                            days = self._days + other.days, 
                            seconds = self._seconds + other.seconds)
        elif isinstance(other, Duration):
            return Duration(years=self._years+other.years, 
                            months=self._months+other.months, 
                            days = self._days + other.days, 
                            seconds = self._seconds + other.seconds)
        elif isinstance(other, datetime):
            newtime = other + timedelta(self.days, self.seconds)
            result_year, result_month = self.normalize_month(newtime.year + self.years, newtime.month + self.months)
            return newtime.replace(year=result_year, month=result_month)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, timedelta):
            return Duration(years=self._years, months=self._months,
                            days = self._days - other.days, 
                            seconds = self._seconds - other.seconds)
        elif isinstance(other, Duration):
            return Duration(years=self._years - other.years, 
                            months=self._months - other.months, 
                            days = self._days - other.days, 
                            seconds = self._seconds - other.seconds)
        elif isinstance(other, datetime):
            newtime = other - timedelta(self.days, self.seconds)
            result_year, result_month = self.normalize_month(newtime.year() - self.years, newtime.month() - self.months)
            return newtime.replace(year=result_year, month=result_month)
        return NotImplemented
    
    def __str__(self):
        s = ""
        if self.seconds > 0:
            mm, ss = divmod(self.seconds, 60)
            hh, mm = divmod(mm, 60)
            s = "%dH%02dN%02dS" % (hh, mm, ss)
        if self.days > 0:
            s = ("%dD" % self.days) + s
        if self.months > 0:
            s = ("%dM" % self.months) + s
        if self.years > 0:
            s = ("%dY" % self.years) + s
        return "P" + s

    def __repr__(self):
        return 'duration(' + self.__str__() + ')'
    
if __name__ == '__main__':
    # 测试星期日期表示法
    test_date = weekday_to_ymd(2020, 4, 3)
    print("%s年第%s周的星期%s是：%s" % (2020, 4, 3, date))
    assert(str(test_date) == "2020-01-22")
    # 测试加减法
    duration = Duration(years = 1, days=5)
    date = date(2018, 12, 29)
    time = datetime(date.year, date.month, date.day)
    print("%s经过%s后是%s" % (date, duration, duration + time))
    