import time

class Animal(object):

    def __init__(self, name):
        print('__init__方法被调用')
        self.__name = name


    #def __del__(self):
    #    print("__del__方法被调用")
    #    print("%s对象马上被干掉了..."%self.__name)


# 创建对象

dog = Animal("哈皮狗")

# 删除对象

del dog

cat = Animal("波斯猫")

cat2 = cat

cat3 = cat

print(id(cat),id(cat2), id(cat3))
import sys
print(sys.getrefcount(cat3))
print("---马上 删除cat对象")

del cat
print(sys.getrefcount(cat3))
print("---马上 删除cat2对象")

del cat2
print(sys.getrefcount(cat3))
print("---马上 删除cat3对象")

del cat3

print("程序2秒钟后结束")

time.sleep(2)