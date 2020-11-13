#! /usr/bin/env python
# coding:utf-8
'''
Created on 2020年1月29日

@author: Lichengang
'''


DATE = 'DATE'
TIME = 'TIME'
RANGE = "RANGE"
DURATION = 'DURATION'

def merge_duration(last : dict, cuurent : dict):
    ''' 合并时间段
    '''
    last["end"] = cuurent["end"]
    last["mention"] = last["mention"] + cuurent["mention"]
    if "result" in last and "result" in cuurent:
        last['result'] = last['result'] + cuurent['result'][1:]
    return last

def merge_date_and_time(last : dict, cuurent : dict):
    ''' 合并时间和日期
    '''
    last["end"] = cuurent["end"]
    last["mention"] = last["mention"] + cuurent["mention"]
    last["type"] = TIME
    if "result" in last and "result" in cuurent:
        last['result'] = last['result'] + cuurent['result']
    return last

def merge(time_list : list):
    ''' 合并时间和日期
    :param time_list 时间片段，list[MatchResult]
    '''
    time_list.sort(key = lambda x : x['start'])
    last_time_exp = None
    new_time_list = []
    for time_exp in time_list:
        merged = False
        attribute = 'type' if 'result' in time_exp else 'subtype'
        if last_time_exp is not None and last_time_exp["end"] == time_exp["start"]:
            if last_time_exp[attribute] == DURATION and time_exp[attribute] == DURATION:
                if merge_duration(last_time_exp, time_exp):
                    merged = True
            if last_time_exp[attribute] == DATE and time_exp[attribute] == TIME:
                if merge_date_and_time(last_time_exp, time_exp):
                    merged = True
        if not merged:
            new_time_list.append(time_exp)
            last_time_exp = time_exp
    return new_time_list

if __name__ == '__main__':
    pass
