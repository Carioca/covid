import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

def solve_sirgm(S_i=210_000_000., I_i=2200., R_i=100., G_i=0, M_i=0,
                beta=0.16, mu=14, gamma=0.01, sigma=0.09, theta=0.01,
                dias=365, N=24):
    """Plot a solution to the SIRGM differential equations."""

    def sir_deriv(sirgm, t0, beta=beta, mu=mu):
        """Compute the time-derivative of a SIRGM system."""
        s, i, r, g, m = sirgm
        n = s + i + r
        return [
            -beta * i * s/n,
            beta * i * s/n - i/mu - gamma*i,
            i/mu + sigma * g,
            gamma * i - sigma * g - theta * g,
            theta * g,
        ]

    t = np.linspace(0,dias,int(N*dias))
    sirgm = integrate.odeint(sir_deriv, (S_i, I_i, R_i, G_i, M_i), t)
    fig,ax = plt.subplots()
    ax.stackplot(t, sirgm[:,1], sirgm[:,3], sirgm[:,4], sirgm[:,0], sirgm[:,2],
                 labels=('Infecciosos', 'Graves', 'Mortos', 'Suscetíveis', 'Recuperados'))
    ax.set_xlabel('Dias')
    ax.set_ylabel('Pessoas')
    ax.set_title(f'Modelo SIRGM, R₀={beta*mu:.3}')
    l = ax.legend()
    print(f'Mortos:{int(sirgm[-1,4]):,}')
    print(f'Demanda por UTI:{int(max(sirgm[:,3])):,}')
    return t, sirgm
