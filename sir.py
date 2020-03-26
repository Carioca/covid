
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import integrate

def solve_sir(S_i=210_000_000., I_i=2200., R_i=100., beta=0.15, mu=14):
    """Plot a solution to the SIR differential equations."""
    max_time = 100.
    N = 24
    
#     fig = plt.figure()
#     ax = fig.add_axes([0, 0, 1, 1])
    
#     ax.set_xlim((0, max_time))
    
    def sir_deriv(s_i_r, t0, beta=beta, mu=mu):
        """Compute the time-derivative of a Lorenz system."""
        s, i, r = s_i_r
        n = s + i + r
        return [
            -beta * i * s/n, 
            beta * i * s/n - i/mu, 
            i/mu
        ]
    
    t = np.linspace(0,max_time,int(N*max_time))
    s_i_r_t = integrate.odeint(sir_deriv, (S_i, I_i, R_i), t)
    plt.plot(t, s_i_r_t)
    plt.show()
    return t, s_i_r_t
