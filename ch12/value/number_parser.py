#! /usr/bin/env python
# coding:utf-8
'''
Created on 2019年11月29日

@author: Lichengang
'''

import re

_CHINESE_LEVELS = {'十':10, '百':100, '千':1000, '拾':10, '佰':100, '仟':1000,
                  '万':10000, '萬':10000, '亿':100000000}
_CHINESE_NUMBERS = {'一':1,'二':2,'三':3,'四':4,'五':5,'两':2,
                   '六':6,'七':7,'八':8,'九':9,'〇':0,
                   '壹':1,'贰':2,'叁':3,'肆':4,'伍':5,
                   '陆':6,'柒':7,'捌':8,'玖':9,'零':0}

level_pattern = re.compile("("+"|".join(_CHINESE_LEVELS.keys())+")")
_thousands = "".join(list(_CHINESE_NUMBERS.keys())) + "十百千拾佰仟"
chn_int_pattern = re.compile("([" + _thousands + "万]+亿)?([" + _thousands + "万]+)")
chn_thousand_pattern = re.compile("(([" + _thousands + "]+)万)?([" + _thousands + "]+)")

def norm_arabic_num(text):
    '''
    · 阿拉伯数字串解析
    :param text:  中文数字正整数穿
    :return int类型
    '''
    match = re.fullmatch("[0-9]+", text)
    return int(text) if match else None

def norm_level(number, valuechar):
    '''
    · 数量级解析和赋值
    :param number:       数字 int
    :param valuechar:    表示级别的字符
    :return int类型
    '''
    return number*_CHINESE_LEVELS[valuechar] if valuechar in _CHINESE_LEVELS else number

def norm_chinese_num(text):
    '''
    · 中文数字解析，最高到万亿级别
    :param text:  中文数字正整数穿
    :return int类型
    '''
    has_level = level_pattern.search(text)
    # 简单数词
    if not has_level:
        return norm_chinese_code(text)
    natched = chn_int_pattern.match(text)
    value = 0
    if not natched:
        return ""
    for chn_yi_part in natched.groups():
        if chn_yi_part is not None:
            if value > 0:
                value = norm_level(value, "亿")
            thousand_matched = chn_thousand_pattern.match(chn_yi_part)
            value += _norm_chinese_section(thousand_matched.group(2)) * 10000 \
                    + _norm_chinese_section(thousand_matched.group(3))
    return value

def _norm_chinese_section(text):
    if text is None or text == "":
        return 0
    value = 0
    base_value = 0
    for char in text:
        if char in _CHINESE_NUMBERS:
            base_value = _CHINESE_NUMBERS[char]
        elif char in _CHINESE_LEVELS:
            if base_value == 0:
                base_value = 1;
            base_value = norm_level(base_value, char);
            value = value + base_value;
            base_value = 0
    value += base_value
    return value

def norm_chinese_code(text):
    '''
    · 中文编码数字（不含数量积词）解析
    :param text:  中文数字正整数穿
    :return int类型
    '''
    value = 0
    for char in text:
        value *= 10
        value += _CHINESE_NUMBERS[char]
    return value

def parse_range(range_pattern : str):
    pattern = r'\[(.+?) ~ (.+?)\]'
    result = {}
    match = re.match(pattern, range_pattern, re.IGNORECASE)
    if match:
        str_begin = match.group(1)
        str_end = match.group(2)
        if str_begin == '-∞':
            result['begin'] = float("-inf")
        elif str_end == '∞':
            result['end'] = float("inf")
        elif str_begin == str_end:    # 表示“左右”
            result['value'] = float(str_begin)
        else:
            result['begin'] = float(str_begin)
            result['end'] = float(str_end)
    return result


def parse(value_list : list):
    for value_item in value_list:
        if 'type' in value_item and value_item['type'] == "NUM":
            value_item['value'] = float(value_item['result'])
        elif 'type' in value_item and value_item['type'] == "RANGE":
            value_item['value'] = parse_range(value_item['result'])
    return value_list


if __name__ == '__main__':
    strict_data_dict = {
            "一": 1,
            "二": 2,
            "十": 10,
            "十一": 11,
            "一十一": 11,
            "一百一十一": 111,
            "一千一百一十一": 1111,
            "一万一千一百一十一": 11111,
            "一十一万一千一百一十一": 111111,
            "一百一十一万一千一百一十一": 1111111,
            "一千一百一十一万一千一百一十一": 11111111,
            "一亿一千一百一十一万一千一百一十一": 111111111,
            "一十一亿一千一百一十一万一千一百一十一": 1111111111,
            "一百一十一亿一千一百一十一万一千一百一十一": 11111111111,
            "一千一百一十一亿一千一百一十一万一千一百一十一": 111111111111,
            "一千一百一十一万一千一百一十一亿一千一百一十一万一千一百一十一": 1111111111111111
        }
    for kay,value in sorted(strict_data_dict.items()):
        result = norm_chinese_num(kay)
        print("%s\t%s" % (result, result == value))







            
