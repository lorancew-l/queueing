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
        'label': r'Реальное распределение',
    },
    'x_label': r'$\rho$',
    'y_label': r'$N(\rho, m)$',
    'analytic': {
        'color': '#3f35d4',
        'alpha': 1.0,
        'linewidth': 2,
        'label': r'Теория',
    },
    'monte_carlo': {
        'color': '#d42086',
        'alpha': 1.0,
        'marker': '8',
        'markersize': 8,
        'linestyle': 'none',
        'label': r'Эксперимент',

    },
    'deviation': {
        'color': '#16adc4',
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


def get_erlang_flow_distribution(left, right, _lambda, r):
    rng = np.random.default_rng()

    n = int(1e5)
    z = np.vectorize(erlang_flow)
    h = np.max(z(np.linspace(left, right, n), _lambda, r))

    res_x = np.array([])

    while True:
        w = rng.random(1)[0] * h

        if len(res_x) < n:
            x = generate_random_numbers(left, right, 40)
            x = x[w < z(x, _lambda, r)]
            res_x = np.concatenate((res_x, x))
        else:
            res_x = res_x[0:n]
            break

    return res_x


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
        a = self.params['a']
        b = self.params['b']
        r = self.params['r']

        x_l = np.linspace(a, b, int(1e5))
        y_l = erlang_flow(x_l, 1, r)
        x = get_erlang_flow_distribution(a, b, 1, r)

        self.axis.plot(x_l, y_l)
        sns.histplot(x, bins=40, ax=self.axis, stat="density",
                     **PLOT_PROPS['hist'])

    def plot_test(self):
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

        step = 0.2

        rng = np.random.default_rng()

        kn = 1
        for L in np.arange(step, b + step, step):
            t = 0
            t_discr = 0
            t_request = 0
            t_service = 0

            k_kurent = 0
            queue = 0

            u = np.array([np.log(rng.random()) for _ in range(r)])

            # TL = -1 / L * np.log(u)
            TL = -1 / (2*L) * np.sum(u)
            TM = - np.sum(u)
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
                        u = rng.random()
                        TM = -np.log(u)
                        t_service = t_request + TM
                    elif queue < m:
                        queue += 1
                        k_kurent += 1
                    u = rng.random()
                    TL = -1 / L * np.log(u)
                    t_request += TL

                if t > t_service and k_kurent > 0:
                    if queue == 0:
                        k_kurent -= 1
                    else:
                        queue -= 1
                        k_kurent -= 1
                        u = rng.random()
                        TM = - np.log(u)
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
