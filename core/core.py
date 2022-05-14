import seaborn as sns
import numpy as np
from matplotlib.pyplot import figure
import math

# виды маркеров https://matplotlib.org/stable/api/markers_api.html
PLOT_PROPS = {
    'hist': {
        'color': '#69e089',
        'alpha': 0.8,
        'edgecolor': '#1ca340',
        'label': r'Распределение на основе равномерного',
    },
    'analytic_hist': {
        'color': '#3f35d4',
        'alpha': 1.0,
        'linewidth': 2,
        'label': r'Теория',
    },
    'x_label': r'$\rho$',
    'y_label': r'$N(\rho, m)$',
    'analytic': {
        'color': '#8732a8',
        'alpha': 1.0,
        'linewidth': 2,
        'label': r'Теория',
    },
    'monte_carlo': {
        'color': '#328da8',
        'alpha': 1.0,
        'marker': '8',
        'markersize': 8,
        'linestyle': 'none',
        'label': r'Эксперимент',

    },
    'deviation': {
        'color': '#32a867',
        'alpha': 1.0,
        'marker': '.',
        'markersize': 12,
        'linestyle': 'none',
        'label': r'Стандартное отклонение',

    },
}


def func(x, m):
    return (x/(1-x)) - (x*(m+2)*x ** (m+1)/(1-x ** (m+2)))


def generate_random_numbers(low, high, size):
    rng = np.random.default_rng()

    numbers = rng.random(size) * (high-low) + low

    return numbers


def erlang_flow(x, _lambda, r):
    return _lambda * np.power(_lambda * x, r - 1) * np.exp(-_lambda * x) / math.factorial(r - 1)


def get_tau(_lambda, r):
    rng = np.random.default_rng()
    return -1 / (_lambda) * np.sum(np.log(np.array([rng.random() for _ in range(r)])))


def get_erlang_flow_distribution(_lambda, r):
    return [get_tau(_lambda, r) for _ in range(100000)]


class Plotter:
    def __init__(self, params=dict(), size=(10, 6)):
        self.figure = figure(figsize=size)
        self.axis = self.figure.add_subplot(111)

        self.set_axis_props()
        self.params = params

        self.plot_stack = []

    def plot(self):
        self.axis.cla()

        for plot_func in self.plot_stack:
            plot_func()

        self.plot_stack = []

        self.axis.legend(frameon=False)
        self.set_axis_props()

        self.figure.canvas.draw_idle()
        self.figure.tight_layout()

    def set_axis_props(self, facecolor='white', tick_color='black', spine_color='black', grid_color='grey', hide_ticks=True):
        self.axis.set_facecolor(facecolor)
        self.axis.tick_params(axis='x', colors=tick_color)
        self.axis.tick_params(axis='y', colors=tick_color)

        self.axis.spines['bottom'].set_color(spine_color)
        self.axis.spines['left'].set_color(spine_color)
        self.axis.spines['top'].set_color(None)
        self.axis.spines['right'].set_color(None)

        if hide_ticks:
            self.axis.tick_params(length=0)

        self.axis.grid(color=grid_color, linestyle='-',
                       linewidth=0.25, alpha=0.6)

    def plot_hist(self):
        r = self.params['r']
        bins = self.params['bins']

        x = get_erlang_flow_distribution(1, r)
        x_l = np.linspace(np.min(x), np.max(x), int(1e5))
        y_l = erlang_flow(x_l, 1, r)

        self.axis.plot(x_l, y_l, **PLOT_PROPS['analytic_hist'])
        sns.histplot(x, bins=bins, ax=self.axis, stat="density",
                     **PLOT_PROPS['hist'])

        self.axis.set_xlabel(r'$\tau$')
        self.axis.set_ylabel(r'$\phi(\tau)$')

    def plot_test(self, mode='ee'):
        a = self.params['a']
        b = self.params['b']
        m = self.params['m']
        N = self.params['N']
        r = self.params['r']

        k = np.zeros(N)
        exper = np.array([])

        s_arr = np.array([])
        theory_x = np.linspace(a, b, int(1e4))
        theory_y = func(theory_x, m)
        L_graph = np.array([])
        experiment_graph_y = np.array([])

        step = 0.3

        rng = np.random.default_rng()

        kn = 1
        for L in np.arange(step, b + step, step):
            t = 0
            t_discr = 0
            t_request = 0
            t_service = 0

            k_kurent = 0
            queue = 0

            u = np.array([rng.random() for _ in range(r)])

            if (mode == 'em'):
                TL = -1 / (L * r) * np.sum(np.log(u))
                TM = - np.log(rng.random())

            if (mode == 'me'):
                TL = -1 / L * np.log(rng.random())
                TM = -1 / r * np.sum(np.log(u))

            if (mode == 'ee'):
                TL = -1 / (L * r) * np.sum(np.log(u))
                TM = -1 / r * np.sum(np.log(u))

            t_request = TL

            t_max = 5 * N
            dt = -1 / L * np.log(0.99)
            i = 0

            while t < t_max:
                if t > t_discr:
                    t_discr += 5
                    k[i] = k_kurent
                    i += 1
                if t > t_request:
                    if k_kurent == 0:
                        k_kurent += 1

                        if (mode == 'em'):
                            TM = - np.log(rng.random())

                        if (mode in ['me', 'ee']):
                            u = np.array([rng.random() for _ in range(r)])
                            TM = -1 / r * np.sum(np.log(u))

                        t_service = t_request + TM
                    elif queue < m:
                        queue += 1
                        k_kurent += 1

                    u = np.array([rng.random() for _ in range(r)])

                    if (mode in ['em', 'ee']):
                        TL = -1 / (L * r) * np.sum(np.log(u))

                    if (mode == 'me'):
                        TL = -1 / L * np.log(rng.random())

                    t_request += TL

                if t > t_service and k_kurent > 0:
                    if queue == 0:
                        k_kurent -= 1
                    else:
                        queue -= 1
                        k_kurent -= 1

                        if (mode == 'em'):
                            TM = - np.log(rng.random())

                        if (mode in ['me', 'ee']):
                            u = np.array([rng.random() for _ in range(r)])
                            TM = -1 / r * np.sum(np.log(u))

                        t_service += TM

                t += dt

            _sum = np.sum(k)

            exper = np.append(exper, _sum / N)
            experiment_graph_y = np.append(experiment_graph_y, exper[kn - 1])

            d = 0

            for i in range(N):
                d += pow(exper[kn - 1] - k[i], 2)

            d = d / (N - 1)

            s = np.sqrt(d)

            s_arr = np.append(s_arr, s)
            L_graph = np.append(L_graph, L)

            kn += 1

        self.axis.plot(theory_x, theory_y, **PLOT_PROPS['analytic'])
        self.axis.plot(L_graph, experiment_graph_y,
                       **PLOT_PROPS['monte_carlo'])
        self.axis.plot(L_graph, s_arr, **PLOT_PROPS['deviation'])

        self.axis.set_xlim(a, b)
        self.axis.set_ylim(0, np.max(theory_y) * 1.1)

        self.axis.set_xlabel(PLOT_PROPS['x_label'])
        self.axis.set_ylabel(PLOT_PROPS['y_label'])
