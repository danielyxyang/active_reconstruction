import contextlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from IPython.display import display


class DynamicPlotter():
    """Class providing support for dynamically updating plots."""
    interactive = False

    @staticmethod
    def set_interactive(interactive=True):
        """Enable or disable interactive plots based on ipyml backend."""
        DynamicPlotter.interactive = interactive

    def __init__(self):
        self.fig = None
        self.axis = None # currently active axis
        self.artists = {}
        self.__displayed = False
    
    def create(self):
        """Create dynamic plot."""
        # create figure
        if self.fig is not None:
            plt.close(self.fig)
        self.fig, self.axis = plt.subplots(constrained_layout=True)
        self.artists = {}
        self.__displayed = False

        # configure figure canvas for ipympl
        if DynamicPlotter.interactive:
            self.fig.canvas.toolbar_position = "top"
            self.fig.canvas.header_visible = False

    def reset(self):
        """Reset dynamic plot."""
        # remove all artists
        self.artists = {}
        self.__displayed = False

    def display(self, out=None, clear=True, rescale=False):
        """Display or redraw plot depending on changes to set of artists."""
        if out is None:
            out = contextlib.nullcontext()
        
        if DynamicPlotter.interactive:
            if self.__displayed:
                # (optional) rescale plot automatically
                if rescale:
                    self.axis.relim()
                    self.axis.autoscale(tight=True)
                # redraw plot
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            else:
                # display plot
                with out:
                    display(self.fig.canvas, clear=clear)
                self.__displayed = True
        else:
            with out:
                display(self.fig, clear=clear)
            self.__displayed = True

    # PLOTTING METHODS

    def static(self, key, plt_f, visible=True):
        """Plot static artist on first call."""
        # display artist
        if key not in self.artists.keys():
            self.artists[key] = plt_f()
        # set visibility of artist
        self.set_visible(self.artists[key], visible)

    def dynamic(self, key, plt_f, update_f, visible=True):
        """Plot dynamic artist on first call and update on later calls."""
        # display or update artist
        if key not in self.artists.keys():
            self.artists[key] = plt_f()
        else:
            update_f(self.artists[key])
        # set visibility of artist
        self.set_visible(self.artists[key], visible)
    
    def dynamic_plot(self, key, *args, visible=True, **kwargs):
        """Plot dynamic Line2D artist on first call and update on later calls."""
        self.dynamic(
            key,
            lambda: self.axis.plot(*args, **kwargs),
            lambda lines: [line.set_data(*args[2*i:2*i+2]) for i, line in enumerate(lines)],
            visible=visible,
        )
    
    def dynamic_patch_collection(self, key, patches, visible=True, **kwargs):
        """Plot dynamic patch collection on first call and update on later calls."""
        self.dynamic(
            key,
            lambda: self.axis.add_collection(PatchCollection(patches, **kwargs)),
            lambda collection: collection.set_paths(patches),
            visible=visible,
        )

    # HELPER METHODS

    def set_visible(self, item, visible):
        """Change visibility of plotted artists."""
        if isinstance(item, plt.Artist):
            item.set_visible(visible)
        elif isinstance(item, list):
            for subitem in item:
                subitem.set_visible(visible)
        else:
            print("WARNING: not able to change visibility of {}".format(item))


class MultipleTicks:
    """Class for formatting multiples of some fractional value in LaTeX as ticks."""
    # reference: https://stackoverflow.com/a/53586826
    
    def __init__(self, denominator=1, number=np.pi, latex="\pi", number_in_frac=True, fracformat=r"\frac{%s}{%s}"):
        """_summary_

        Args:
            denominator (float, optional): Number of ticks between integer
                multiples of `number`. Defaults to 1.
            number (float, optional): Numeric value of `latex`. Defaults to
                np.pi.
            latex (str, optional): LaTeX string of `number`. Defaults to "\pi".
            number_in_frac (bool, optional): Flag whether `latex` string should
                included in numerator of fraction or outside. Defaults to True.
            fracformat (str, optional): LaTeX format for fraction with first %s
                replaced by numerator and second %s by denominator. Defaults to
                r"\frac{%s}{%s}".
        """        
        self.denominator = denominator
        self.number = number
        self.latex = latex
        self.number_in_frac = number_in_frac
        self.fracformat = fracformat
    
    def scalar_formatter(self, scalar):
        """Format scalar value."""
        if scalar == 0:
            return "$0$"
        if scalar == 1:
            return "${}$".format(self.latex)
        elif scalar == -1:
            return "$-{}$".format(self.latex)
        else:
            return "${}{}$".format(scalar, self.latex)
    
    def fraction_formatter(self, num, den):
        """Format fractional value."""
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
        """Format value as scalar or fraction."""
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
        """Return matplotlib locator."""
        if self.denominator <= 1:
            scalar = int(np.rint(1 / self.denominator))
            return plt.MultipleLocator(scalar * self.number)
        else:
            return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        """Return matplotlib formatter."""
        return plt.FuncFormatter(self.multiple_formatter)
