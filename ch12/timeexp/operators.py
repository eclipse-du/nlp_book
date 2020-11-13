#! /usr/bin/env python
# coding:utf-8
'''
Created on 2012年12月5日
@copyright: code of chapter 12
@author: Lichengang
'''

from framework.entity_parser import ParseKnowledgeBase, FunctionOperator
from value import number_parser

def create_config():
    config = {}
    # 两位数字在大于多少后指的是上一个世纪（20世纪）
    config['two_digits_max_year'] = 50
    # 是否包含结尾时间，如“3月”的结束时间到4月1日0时
    config['time_include_end'] = True
    # 倾向于推理出未来的时间
    config['prefer_future'] = True
    return config

def parse_number(text, count):
    value = number_parser.norm_arabic_num(text)
    if not value:
        value = number_parser.norm_chinese_num(text)
    format = '%%0%sd' % count
    return format % value

def morm_number(operand : list, config : dict):
    ''' 解析数值，operand[0]为数值，perand[1]为最少位数
    '''
    assert len(operand) >= 1
    value = parse_number(operand[0], int(operand[1]))
    return str(value)

def morm_digit(operand : list, config : dict):
    ''' 解析无单位的数值，operand[0]为数值
    '''
    assert len(operand) >= 1
    value = number_parser.norm_arabic_num(operand[0])
    if not value:
        value = number_parser.norm_chinese_code(operand[0])
    return str(value)
    
def morm_century(operand : list, config : dict):
    ''' 处理“世纪”未知的情况
    '''
    year_two_digit = parse_number(operand[0], 2)
    year_two_digit = int(year_two_digit)
    base = 1900
    if year_two_digit < config['two_digits_max_year']:
        base += 100
    return str(base + year_two_digit)

def add_operator(operand : list, config : dict):
    value = int(operand[0]) + int(operand[1])
    return str(value)

def future_past(operand : list, config : dict):
    ''' 根据配置，决定“一个月内”是向前还是向后
    '''
    return "LLLL-LL-LL/" if config['prefer_future'] else "/LLLL-LL-LL"

def to_explict_time(operand : list, config : dict):
    ''' 12/24小时制换算
    '''
    assert len(operand) >= 2
    value = int(parse_number(operand[1], 1))
    if operand[0] == 'am' and value > 12 and value <= 23:
        value -= 12
    if operand[0] == 'noon' and value < 3:
        value += 12
    if operand[0] == 'pm' and value < 12:
        value += 12
    if operand[0] == 'midnight':
        if value >= 12 and value <= 18:
            value -= 12
        elif value >= 6 and value < 12:
            value += 12;
    return str(value)

def expand_time_range(operand : list, config):
    ''' 展开时间的时间段
    ''' 
    assert len(operand) >= 1
    part_list = operand[0].split("/")
    return "LLLL-LL-LLT" + part_list[0] + "/LLLL-LL-LLT" + part_list[1]

def init(parser : ParseKnowledgeBase):
    parser.add_operator(FunctionOperator("NormDigit", morm_digit))
    parser.add_operator(FunctionOperator("NormNumber", morm_number))
    parser.add_operator(FunctionOperator("NormCentury", morm_century))
    parser.add_operator(FunctionOperator("FutureOrPast", future_past))
    parser.add_operator(FunctionOperator("add", add_operator))
    parser.add_operator(FunctionOperator("ToDayInHour", to_explict_time))
    parser.add_operator(FunctionOperator("ExpandTimeRange", expand_time_range))

if __name__ == '__main__':
    config = create_config()
    print(morm_century(['零八'], config))
    print(add_operator(['3', '1'], config))
    print(expand_time_range(['18:00:00/22:00:00'], config))