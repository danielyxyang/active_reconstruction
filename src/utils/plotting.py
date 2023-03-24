import numpy as np
import matplotlib.pyplot as plt


class MultipleTicks:
    # reference: https://stackoverflow.com/a/53586826
    
    def __init__(self, denominator=None, number=np.pi, latex="\pi", number_in_frac=True):
        """
        Set a tick on each integer multiple of `number` if `denominator` is None
        or every 1/`denominator`-th multiple of `number`.
        """
        self.denominator = denominator
        self.number = number
        self.latex = latex
        self.number_in_frac = number_in_frac
    
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
                return "$\\frac{%s}{%s}$" % (self.latex, den)
            elif num == -1:
                return "$-\\frac{%s}{%s}$" % (self.latex, den)
            elif num < -1:
                return "$-\\frac{%s%s}{%s}$" % (-num, self.latex, den)
            else:
                return "$\\frac{%s%s}{%s}$" % (num, self.latex, den)
        else:
            if num < 0:
                return "$-\\frac{%s}{%s}%s$" % (-num, den, self.latex)
            else:
                return "$\\frac{%s}{%s}%s$" % (num, den, self.latex)
    
    def multiple_formatter(self, x, pos):
        if self.denominator is None:
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
        if self.denominator is None:
            return plt.MultipleLocator(self.number)
        else:
            return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(self.multiple_formatter)
