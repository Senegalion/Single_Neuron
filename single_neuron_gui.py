import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ==========================
# 1. Neuron z różnymi aktywacjami
# ==========================

class SingleNeuron:
    def __init__(self, n_inputs: int, activation: str = "heaviside",
                 train_activation: str = "sigmoid",
                 lr: float = 0.1, beta: float = 1.0):
        """
        n_inputs      - liczba wejść (bez biasu)
        activation    - funkcja do EWALUACJI (heaviside/sigmoid/sin/tanh/sign/relu/lrelu)
        train_activation - funkcja używana w UCZENIU ("heaviside" lub "sigmoid")
        lr            - learning rate
        beta          - parametr sigmoidy
        """
        self.n_inputs = n_inputs
        self.activation_name = activation
        self.train_activation_name = train_activation
        self.lr = lr
        self.beta = beta

        # wagi + bias w jednym wektorze (ostatni element = bias)
        self.w = np.random.randn(n_inputs + 1) * 0.1

    # ----- aktywacje -----
    def _f(self, s, name=None):
        if name is None:
            name = self.activation_name

        if name == "heaviside":
            return np.where(s >= 0.0, 1.0, 0.0)
        if name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-self.beta * s))
        if name == "sin":
            return np.sin(s)
        if name == "tanh":
            return np.tanh(s)
        if name == "sign":
            return np.sign(s)
        if name == "relu":
            return np.where(s > 0.0, s, 0.0)
        if name == "lrelu":
            return np.where(s > 0.0, s, 0.01 * s)
        raise ValueError(f"Unknown activation {name}")

    def _f_prime(self, s):
        """Pochodna funkcji aktywacji DO TRENINGU.
        Wspieramy Heaviside (założenie f' = 1) i sigmoid.
        """
        if self.train_activation_name == "heaviside":
            # wykład mówi: załóżmy, że pochodna = 1
            return np.ones_like(s)
        if self.train_activation_name == "sigmoid":
            y = self._f(s, name="sigmoid")
            return self.beta * y * (1.0 - y)
        raise ValueError(f"Training derivative not defined for {self.train_activation_name}")

    # ----- pomocnicze -----
    def _net_input(self, X):
        # X: (N, n_inputs)
        # dodajemy bias = 1
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        return Xb @ self.w, Xb

    # ----- predykcja -----
    def predict_proba(self, X):
        s, _ = self._net_input(X)
        return self._f(s)    # ciągłe wyjście

    def predict_label(self, X):
        y = self.predict_proba(X)
        # klasy 0/1
        return (y >= 0.5).astype(int)

    # ----- trening -----
    def train(self, X, d, epochs: int = 50, shuffle: bool = True):
        """
        X : (N, n_inputs)
        d : (N,) etykiety 0/1
        """
        X = np.asarray(X, dtype=float)
        d = np.asarray(d, dtype=float).flatten()

        n_samples = X.shape[0]

        for ep in range(epochs):
            if shuffle:
                idx = np.random.permutation(n_samples)
                X = X[idx]
                d = d[idx]

            for j in range(n_samples):
                xj = X[j:j+1]  # (1, n_inputs)
                target = d[j]

                s, xj_b = self._net_input(xj)  # s: (1,), xj_b: (1, n_inputs+1)
                s = s[0]
                xj_b = xj_b[0]

                # wyjście DO TRENINGU – używamy train_activation
                y_train = self._f(s, name=self.train_activation_name)
                error = target - y_train
                grad = self._f_prime(s) * error

                # aktualizacja wag
                self.w += self.lr * grad * xj_b


# ==========================
# 2. Generator danych 2D (Gaussy)
# ==========================

def generate_gaussian_class(n_modes: int, samples_per_mode: int,
                            mean_range=(-1.0, 1.0), std_range=(0.1, 0.3), seed=None):
    if seed is not None:
        np.random.seed(seed)

    all_samples = []
    for _ in range(n_modes):
        mx = np.random.uniform(*mean_range)
        my = np.random.uniform(*mean_range)
        sx = np.random.uniform(*std_range)
        sy = np.random.uniform(*std_range)
        cov = np.diag([sx**2, sy**2])

        samples = np.random.multivariate_normal([mx, my], cov, samples_per_mode)
        all_samples.append(samples)

    return np.vstack(all_samples)  # (n_modes*samples_per_mode, 2)


def generate_two_class_data(modes_c0: int, modes_c1: int, samples_per_mode: int):
    X0 = generate_gaussian_class(modes_c0, samples_per_mode)
    X1 = generate_gaussian_class(modes_c1, samples_per_mode)

    y0 = np.zeros(X0.shape[0], dtype=int)
    y1 = np.ones(X1.shape[0], dtype=int)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    return X, y


# ==========================
# 3. GUI + wizualizacja
# ==========================

class NeuronGUI:
    def __init__(self, master):
        self.master = master
        master.title("Single Neuron Classifier")

        # dane
        self.X = None
        self.y = None
        self.neuron = None

        # --- panel kontroli ---
        control_frame = ttk.Frame(master)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(control_frame, text="Modes class 0:").pack(anchor=tk.W)
        self.modes_c0_var = tk.IntVar(value=2)
        ttk.Entry(control_frame, textvariable=self.modes_c0_var, width=6).pack(anchor=tk.W)

        ttk.Label(control_frame, text="Modes class 1:").pack(anchor=tk.W)
        self.modes_c1_var = tk.IntVar(value=2)
        ttk.Entry(control_frame, textvariable=self.modes_c1_var, width=6).pack(anchor=tk.W)

        ttk.Label(control_frame, text="Samples per mode:").pack(anchor=tk.W)
        self.samples_var = tk.IntVar(value=50)
        ttk.Entry(control_frame, textvariable=self.samples_var, width=6).pack(anchor=tk.W)

        ttk.Label(control_frame, text="LR (eta):").pack(anchor=tk.W)
        self.lr_var = tk.DoubleVar(value=0.1)
        ttk.Entry(control_frame, textvariable=self.lr_var, width=6).pack(anchor=tk.W)

        ttk.Label(control_frame, text="Epochs:").pack(anchor=tk.W)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Entry(control_frame, textvariable=self.epochs_var, width=6).pack(anchor=tk.W)

        ttk.Label(control_frame, text="Train activation:").pack(anchor=tk.W)
        self.train_act_var = tk.StringVar(value="sigmoid")
        ttk.Combobox(control_frame, textvariable=self.train_act_var,
                     values=["heaviside", "sigmoid"], width=10, state="readonly").pack(anchor=tk.W)

        ttk.Label(control_frame, text="Eval activation:").pack(anchor=tk.W)
        self.eval_act_var = tk.StringVar(value="heaviside")
        ttk.Combobox(control_frame, textvariable=self.eval_act_var,
                     values=["heaviside", "sigmoid", "sin", "tanh",
                             "sign", "relu", "lrelu"],
                     width=10, state="readonly").pack(anchor=tk.W)

        ttk.Button(control_frame, text="Generate data", command=self.on_generate).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Train neuron", command=self.on_train).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Plot decision boundary", command=self.on_plot).pack(fill=tk.X, pady=5)

        self.info_label = ttk.Label(control_frame, text="No data yet")
        self.info_label.pack(anchor=tk.W, pady=10)

        # --- matplotlib figure ---
        fig, ax = plt.subplots(figsize=(5, 5))
        self.fig = fig
        self.ax = ax
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # ----- actions -----

    def on_generate(self):
        modes_c0 = self.modes_c0_var.get()
        modes_c1 = self.modes_c1_var.get()
        samples = self.samples_var.get()

        self.X, self.y = generate_two_class_data(modes_c0, modes_c1, samples)
        self.info_label.config(text=f"Generated {len(self.y)} samples")

        self._plot_points_only()

    def on_train(self):
        if self.X is None:
            self.info_label.config(text="Generate data first!")
            return

        lr = self.lr_var.get()
        epochs = self.epochs_var.get()
        train_act = self.train_act_var.get()
        eval_act = self.eval_act_var.get()

        self.neuron = SingleNeuron(
            n_inputs=2,
            activation=eval_act,
            train_activation=train_act,
            lr=lr,
            beta=1.0
        )
        self.neuron.train(self.X, self.y, epochs=epochs)
        self.info_label.config(text=f"Trained neuron ({epochs} epochs)")

    def on_plot(self):
        if self.X is None or self.neuron is None:
            self.info_label.config(text="Need data + trained neuron")
            return

        self._plot_with_boundary()

    # ----- plotting helpers -----

    def _plot_points_only(self):
        self.ax.clear()
        if self.X is not None:
            idx0 = self.y == 0
            idx1 = self.y == 1
            self.ax.scatter(self.X[idx0, 0], self.X[idx0, 1], c="blue", label="class 0")
            self.ax.scatter(self.X[idx1, 0], self.X[idx1, 1], c="red", label="class 1")
            self.ax.legend()
        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        self.ax.set_title("Data samples")
        self.ax.grid(True)
        self.canvas.draw()

    def _plot_with_boundary(self):
        self.ax.clear()

        # zakres siatki
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = self.neuron.predict_label(grid)
        zz = zz.reshape(xx.shape)

        # tło – dwa kolory dla dwóch klas
        self.ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5],
                         colors=["#ccddff", "#ffcccc"], alpha=0.7)

        # punkty treningowe
        idx0 = self.y == 0
        idx1 = self.y == 1
        self.ax.scatter(self.X[idx0, 0], self.X[idx0, 1], c="blue", edgecolor="k", label="class 0")
        self.ax.scatter(self.X[idx1, 0], self.X[idx1, 1], c="red", edgecolor="k", label="class 1")

        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        self.ax.set_title(f"Decision boundary ({self.neuron.activation_name})")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuronGUI(root)
    root.mainloop()