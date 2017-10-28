import matplotlib as mpl
mpl.pyplot.Axes.scatter
import numpy as np

def plot_points(x, t, axes):
    return axes.scatter(x, t, marker='o',c='#ffffff00', linewidths=2, 
                        edgecolors='b', zorder=10, s=5)

def plot_dots(x, t, axes):
    return axes.scatter(x, t, marker='.', c='k', zorder=10, alpha=0.2, s=0.2)
    
    
def plot_line(x, t, axes):
    return axes.plot(x, t, c='k', alpha=0.3)
    
def plot_probability(prob_grid, axes, extent):
    return axes.imshow(prob_grid, cmap='RdYlGn', interpolation='gaussian', alpha=0.8,
                extent = extent, origin="lower")