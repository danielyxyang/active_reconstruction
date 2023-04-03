import numpy as np
import matplotlib.pyplot as plt


class MultipleTicks:
    # reference: https://stackoverflow.com/a/53586826
    
    def __init__(self, denominator=1, number=np.pi, latex="\pi", number_in_frac=True, fracformat=r"\frac{%s}{%s}"):
        """
        Set `denominator` many ticks between integer multiples of `number`.
        """
        self.denominator = denominator
        self.number = number
        self.latex = latex
        self.number_in_frac = number_in_frac
        self.fracformat = fracformat
    
    def scalar_formatter(self, scalar):
        if scalar == 0:
            return "$0$"
        if scalar == 1:
            return "${}$".format(self.latex)
        elif scalar == -1:
            return "$-{}$".format(self.latex)
        else:
            return "${}{}$".format(scalar, self.latex)
    
    def fraction_formatter(self, num, den):
        if self.number_in_frac:
            if num == 1:
                return "${}$".format(self.fracformat % (self.latex, den))
            elif num == -1:
                return "$-{}$".format(self.fracformat % (self.latex, den))
            elif num < -1:
                return "$-{}$".format(self.fracformat % (str(-num) + self.latex, den))
            else:
                return "${}$".format(self.fracformat % (str(num) + self.latex, den))
        else:
            if num < 0:
                return "$-{}{}$".format(self.fracformat % (-num, den), self.latex)
            else:
                return "${}{}$".format(self.fracformat % (num, den), self.latex)
    
    def multiple_formatter(self, x, pos):
        if self.denominator <= 1:
            scalar = int(np.rint(x / self.number))
            return self.scalar_formatter(scalar)
        else:
            # cancel gcd
            den = self.denominator
            num = int(np.rint(x * den / self.number))
            gcd = np.gcd(num, den)
            num, den = int(num / gcd), int(den / gcd)
            # format fraction
            if den == 1:
                return self.scalar_formatter(num)
            else:
                return self.fraction_formatter(num, den)

    def locator(self):
        if self.denominator <= 1:
            scalar = int(np.rint(1 / self.denominator))
            return plt.MultipleLocator(scalar * self.number)
        else:
            return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(self.multiple_formatter)
