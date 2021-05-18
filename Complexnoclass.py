class ComplexNumber:
    def __init__(self,real,imag):
        self.real=real
        self.imag=imag
    def add(self,complex_no):
        self.real = self.real+complex_no.real
        self.imag = self.imag+complex_no.imag
    def substract(self,c):
        self.real = self.real - c.real
        self.imag = self.imag - c.imag
    def __str__(self):
        return (str(self.real)+"+i"+str(self.imag))
c1 = ComplexNumber(6,12)
c2 = ComplexNumber(10,19)

c1.add(c2)
print(c1.real)
print(c1.imag)
c1.substract(c2)
print(c1)
