import os

import matplotlib.pyplot as plt
import numpy as np

from tqdm.autonotebook import tqdm
from IPython import display


class LossTracker:

    def __init__(self, total):

        self.total = total

        self.pbar = None
        self.hdisplay = None
        self.fig = None
        self.ax = None
        self.line = None

        self.hist_loss = []

        self.i = 0

    def __enter__(self):

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.hdisplay = display.display(None, display_id=True)
        self.pbar = tqdm(total=self.total)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.close()
        plt.close(self.fig)

    def append(self, loss):
        self.hist_loss.append(loss)

    def update(self, pb_only=False, n_display=None):

        if not pb_only:
            x = np.arange(len(self.hist_loss))
            y = self.hist_loss
            n = len(x)
            if n_display is not None and n > n_display:
                step = n // n_display
                x = x[::step]
                y = y[::step]
            self.line.set_xdata(x)
            self.line.set_ydata(y)

            self.ax.relim()
            self.ax.autoscale_view()

            self.hdisplay.update(self.fig)

        self.pbar.update()
        self.pbar.set_postfix({'loss': f'{self.hist_loss[-1]:.3f}'})


if __name__ == "__main__":

    epochs = 100
    with LossTracker(epochs) as dp:

        for i in range(epochs):

            for j in range(50):
                dp.append(np.random.random())

            dp.update()
