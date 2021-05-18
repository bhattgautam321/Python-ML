#Write a class triangle three characterics a b c

import sys
import math
a = float(input("Enter side a = "))
b = float(input("Enter side b = "))
c = float(input("Enter side c = "))
class triangle():
    def __init__(self,a,b,c):
       self.a = a
       self.b = b
       self.c = c
    def area(self):
        s=(a + b + c)/2
        print("Semiperimeter of Triangle =",s)
        area=math.sqrt(s*(s-a)*(s-b)*(s-c))
        return area
t = triangle(a, b, c)
print("Area of Triangle = ",t.area())
