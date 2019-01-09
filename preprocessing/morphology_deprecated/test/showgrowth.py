#coding=utf-8

# -*- coding: utf-8 -*-
import objgraph

_cache = []

class OBJ(object):
    pass

def func_to_leak():
    o  = OBJ()
    _cache.append(o)
    # do something with o, then remove it from _cache

    if True: # this seem ugly, but it always exists
        return
    _cache.remove(o)

if __name__ == '__main__':
    objgraph.show_growth()
    try:
        func_to_leak()
    except:
        pass
    print('after call func_to_leak')
    objgraph.show_growth()

    del _cache

    objgraph.show_growth()
