class CouponBonds: 

    def __init__(self, principal, rate, maturity, interest_rate):
        self.principal = principal
        self.rate = rate
        self.maturity = maturity
        self.interest_rate = interest_rate/100
    
    def current_value(self, x, n):
        return (x / (1+self.interest_rate)**n)
    
if __name__ == '__main__':

    bond = CouponBonds(1000, 10, 3, 4)
    price = 0 

    for t in range(1, bond.maturity+1):
        price = price + bond.current_value(bond.rate*bond.principal/100, t)
        # print(price)
    
    price = price + bond.current_value(bond.principal, bond.maturity)

    print("The current value of the bond is", round(price, 2))