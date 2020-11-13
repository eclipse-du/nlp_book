#! /usr/bin/env python
# coding:utf-8
'''
Created on 2019年11月27日
@copyright: code of chapter 12
@author: Lichengang
'''
import logging
from genericpath import exists

logger = logging.getLogger(__name__)
        
class WordGroup(object):
    '''
    · 词库中的条目
    '''
    
    def __init__(self, name, value):
        # 名称
        self.name = name
        # 词表
        self.word_list = value
        # 词库形成的正则
        self.pattern = ""
 
class ParserRule(object):
    '''
    · 实体解析的单独规则，
    · 每条规则包含匹配部分和执行部分
    '''

    def __init__(self, name, type, pattern_string):
        '''
        Constructor
        '''
        # 名称
        self.name = name
        # 子类型
        self.type = type
        # 匹配模式
        self.pattern_string = pattern_string
        # 形成的正则表达式
        self.regex = None
        # 正则分组计数
        self.parenthes_count = 0
        # 正则分组对应关系
        self.group_mapping = []
        # 操作符
        self.operations = []
        self.parenthes_count = 0
  
class Operation(object):
    '''
    · 操作规则类，包含操作符，操作数引用，结果名字
    '''
    def __init__(self, operator):
        self.operator = operator
        self.operand = []
        self.result = ""
    
    def __str__(self):
        return "Operation %s = %s(%s)" % (self.result, self.operator, self.operand)

    def __repr__(self):
        return self.__str__()

class Operator(object):
    '''
    · 算子的基类
    '''

    def __init__(self, name):
        ''' Constructor
        :param    name 算子的名字
        '''
        self.name = name
        
    def operate(self, *args, **kvargs):
        pass

class FunctionOperator(Operator):
    '''
    · 函数式算子，将函数当作对象包装，以方便外部调用
    '''

    def __init__(self, name, func):
        ''' Constructor
        :param    name 算子的名字
        :param    func 执行的方法
        '''
        Operator.__init__(self, name)
        self.func = func
        
    def operate(self, *args, **kvargs):
        return self.func(*args, **kvargs)

class ReplaceOperator(Operator):
    '''
    · 替换式算子，每个算子内部有一个字典，
    · 接受一个输入参数，输出一个替换后结果
    '''

    def __init__(self, name, replace_dic):
        ''' Constructor
        :param    name 算子的名字
        :param    replace_dic 替换字典
        '''
        Operator.__init__(self, name)
        self.replace_dict = replace_dic
        
    def operate(self, operand : list, config):
        ''' 执行替换式算子
        :param    operand 算子的参数列表
        :param    config 配置项
        '''
        key = ""
        if len(operand) > 0:
            key = operand[0]
        if key not in self.replace_dict:
            return key
        return self.replace_dict[key]


class ParseKnowledgeBase(object):
    '''
    · 实体解析知识库， 包含解析规则和操作符两个部分
    '''

    def __init__(self, type, lang="general", config={}):
        self.type = type
        self.operators = {}
        self.rules = {}
        self.config = config
     
    def add_rule(self, rule):
        self.rules[rule.name] = rule

    def add_operator(self, operator):
        self.operators[operator.name] = operator
        
    def verify_operation(self, operation_List : list):
        ''' 验证规则，确保其中对操作符可以使用
        '''
        is_ok = True
        for opeartion_part in operation_List:
            if isinstance(opeartion_part, list):
                for operation in opeartion_part:
                    if operation.operator == "group":
                        if len(operation.operand) < 1:
                            is_ok = False
                            break
                    elif operation.operator not in self.operators:
                        logger.warning("operator %s not found" % operation.operator)
                        is_ok = False
                        break
        return is_ok

    def _do_operations(self, operation_list : list, match_info):
        ''' 执行规则列表
        :return result as a string 
        '''
        variables = {}  # 用于保存变量的字典
        result = None
        for operation in operation_list:
            if isinstance(operation, Operation):
                if operation.operator == "group":
                    group_id = int(operation.operand[0])
                    result = match_info.group(group_id)
                    if not result:
                        logger.warning("%s group %s not found." % (match_info.re, group_id))
                else:
                    operand_list = [str(variables[operand]) if operand.startswith("__var_") else operand
                                     for operand in operation.operand]
                    operator = self.operators[operation.operator]
                    result = operator.operate(operand_list, self.config)
                variables[operation.result] = result
        return str(result)
    
    def inference(self, match, rule):
        ''' 根据规则，对匹配结果或作推导
        :param     match 正则匹配结果
        :param     rule 规则
        :return:   实体中间结果字符串
        '''
        final_list = [self._do_operations(item, match) if isinstance(item, list) else item
                      for item in rule.operations]
        return "".join(final_list)
