class BankAccount:
    def __init__(self,name,Id):
        self.person=name
        self.accno=Id
        self.balance=0
    def deposit(self,money):
        self.balance=self.balance+money
    def withdraw(self,money):
        if (money<self.balance):
            self.balance=self.balance-money
        else:
            print("Insufficient fund for this transaction")
        
