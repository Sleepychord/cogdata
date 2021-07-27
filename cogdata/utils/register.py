# -*- encoding: utf-8 -*-
'''
@File    :   register.py
@Time    :   2021/07/15 21:21:49
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

ALLCLASSES = {}


def register(cls):
    if hasattr(cls, 'REGISTER_NAME'):
        name = cls.REGISTER_NAME
    else:
        name = cls.__name__
    assert name not in ALLCLASSES
    ALLCLASSES[name] = cls
    return cls
