import matplotlib
matplotlib.use('AGG')


import lin_model as lm
import data_source as ds
import plotter as plr

import numpy as np

import matplotlib.pyplot as plt

sublen=7500
lm1 = lm.LinearModel(lm.gaussian_features, 24)
(X, T) = ds.get_points(sublen, 
        x=np.random.uniform(low=0, high=2*np.pi, size=sublen))

print(X.shape, T.shape)

xrange = np.linspace(0, np.pi*2, num=100)
xrange.shape = (100, 1)
trange = np.linspace(-3, 3, num=100)
trange.shape = (100, 1)




observed_x = []
observed_t = []
fig, axes = plt.subplots(figsize=(7, 6))
i = 0

for (x_p, t_p) in zip(X, T):

    axes.clear()   
    
    prob_grid = np.concatenate(
        [lm1.probability([x], trange) for x in xrange], axis=1)
        
    plr.plot_probability(prob_grid, axes, (0, np.pi * 2, -3, 3))
    plr.plot_line(xrange, ds.get_clean_data(xrange), axes)
    

    for _ in range(20):
        w = lm1.draw_W()
        t = lm1.model_value(xrange, w)
        plr.plot_line(xrange, t, axes)
            
    plr.plot_line(xrange, ds.get_clean_data(xrange), axes)
    plr.plot_points(observed_x, observed_t, axes)
    
    axes.set_xlim([0, np.pi*2])
    axes.set_ylim([-3, 3])
    
    i = i + 1
    lm1.observe(np.array([x_p]), np.array([t_p]))
    observed_x.append(x_p)
    observed_t.append(t_p)
 
    fig.savefig(r'figures/{}.png'.format(i))
    