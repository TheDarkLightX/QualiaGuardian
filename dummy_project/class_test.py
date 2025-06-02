# Test file for large class detection

class SmallClass: # 2 methods
    def method_a(self):
        pass
    def method_b(self):
        pass

class BigClassRegularMethods: # 11 methods
    def m1(self): pass
    def m2(self): pass
    def m3(self): pass
    def m4(self): pass
    def m5(self): pass
    def m6(self): pass
    def m7(self): pass
    def m8(self): pass
    def m9(self): pass
    def m10(self): pass
    def m11_this_makes_it_eleven(self): pass

class AnotherSmallClass:
    """This class has one method."""
    def __init__(self):
        self.data = "initialized"

    def process(self):
        print(self.data)