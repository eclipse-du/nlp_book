import entityparse

if __name__ == '__main__':
    print(entityparse.tag_entity("1小时15分钟是一又四分之一个小时"))
    print(entityparse.tag_entity("会议从十一月二十五日下午三点半开始，持续三个小时"))
    print(entityparse.tag_and_parse_number("3650多万元"))
    print(entityparse.tag_and_parse_number("1小时15分钟是一又四分之一个小时"))
    print(entityparse.tag_and_parse_time("1小时15分钟是一又四分之一个小时"))
    print(entityparse.tag_and_parse_time("会议从十一月二十五日下午三点半开始，持续三个小时"))