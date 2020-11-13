# 第十二章 实体语义理解

entityparse.py 为主模块

## 使用方法
'''python

    import entityparse
    entityparse.tag_entity("明天上午八点")
'''

- `tag_entity` : 利用内置的规则抽取实体，目前包括了“数值”和“时间”两种
- `tag_and_parse_entity` : 抽取并解析实体，目前包括了“数值”和“时间”两种
- `tag_and_parse_number` : 抽取并解析数字
- `tag_and_parse_time` : 抽取并解析时间

## 规则维护

可以通过维护规则库，增加定制的规则

- res/time/general.json   所有语言共通的时间抽取规则库
- res/time/chinsse.json   中文的时间抽取规则库
- res/value/general.json  所有语言共通的数值抽取规则库
- res/value/chinsse.json  中文的数值抽取规则库