import matplotlib.pyplot as plt
import numpy as np

from tqdm.autonotebook import tqdm
from IPython import display

plt.ion()


class LossTracker:

    def __init__(self, total, online_plot=False,
                 online_plot_freq_update=None,
                 online_plot_max_n=None):

        if online_plot:
            if online_plot_freq_update is None:
                online_plot_freq_update = 50
            if online_plot_max_n is None:
                online_plot_max_n = 1000

        self.total = total
        self.online_plot = online_plot
        self.online_plot_freq_update = online_plot_freq_update
        self.online_plot_max_n = online_plot_max_n

        self.pbar = None
        self.hdisplay = None
        self.fig = None
        self.ax = None
        self.line = None

        self.hist_loss = []
        self.i = 0

    def __enter__(self):

        if self.online_plot:
            self.fig, self.ax = plt.subplots()
            self.line, = self.ax.plot([], [])
            self.hdisplay = display.display((), display_id=True)
            if self.hdisplay is None:
                # not using notebook
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        self.pbar = tqdm(total=self.total)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.online_plot:
            plt.close(self.fig)
        self.pbar.close()

    def append(self, loss):
        self.hist_loss.append(loss)

    def update(self):

        if self.online_plot \
                and self.i > 0 \
                and self.i % self.online_plot_freq_update == 0:

            x = np.arange(len(self.hist_loss))
            y = self.hist_loss
            n = len(x)
            if n > self.online_plot_max_n:
                step = n // self.online_plot_max_n
                x = x[::step]
                y = y[::step]
            self.line.set_xdata(x)
            self.line.set_ydata(y)

            self.ax.relim()
            self.ax.autoscale_view()

            if self.hdisplay is not None:
                self.hdisplay.update(self.fig)
            else:
                try:
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                except:
                    pass

        self.pbar.update()
        self.pbar.set_postfix({'loss': f'{self.hist_loss[-1]:.3f}'})

        self.i += 1


if __name__ == "__main__":

    import time

    epochs = 100
    with LossTracker(total=epochs,
                     online_plot=True,
                     online_plot_max_n=np.inf,
                     online_plot_freq_update=1) as dp:

        for i in range(epochs):

            for j in range(50):
                dp.append(np.random.random())

            dp.update()
            time.sleep(0.1)
