n = int(input("Enter the no. = "))    #  153
sum=0
temp= n                             #153             
while temp>0:                       #True            True            True            False (Loop exit)
    digit = temp%10                 #153%10=3        15%10=5         1%10 = 1
    sum += digit**3                 #3^3 = 27        5^3 = 125       1^3 = 1
    temp//=10                       #153//10 = 15    15//10 = 1      1//10 = 0
if (n==sum):
    print("Number",n ,"is Armstrong number")
else:
    print("Not an Armstrong number")
