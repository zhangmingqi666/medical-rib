# -*- coding: utf-8 -*-
import objgraph, sys
class OBJ(object):
    pass

def show_direct_cycle_reference():
    a = OBJ()
    a.attr = a
    objgraph.show_backrefs(a, max_depth=5, filename = "direct.dot")

def show_indirect_cycle_reference():
    a, b = OBJ(), OBJ()
    a.attr_b = b
    b.attr_a = a
    objgraph.show_backrefs(a, max_depth=5, filename = "indirect.dot")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        show_direct_cycle_reference()
    else:
        show_indirect_cycle_reference()
