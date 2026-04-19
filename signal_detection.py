from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

class SignalDetection:
    def __init__(self, hits, misses, false_alarms, correct_rejections):
        self.hits = hits
        self.misses = misses
        self.false_alarms = false_alarms
        self.correct_rejections = correct_rejections

    def hit_rate(self):
        return self.hits / (self.hits + self.misses)

    def false_alarm_rate(self):
        return self.false_alarms / (self.false_alarms + self.correct_rejections)
    
    def d_prime(self):
        return norm.ppf(self.hit_rate()) - norm.ppf(self.false_alarm_rate())
    
    def criterion(self):
        return -0.5 * (norm.ppf(self.hit_rate()) + norm.ppf(self.false_alarm_rate()))
    
    def __add__(self, other):
        return SignalDetection(
            self.hits + other.hits,
            self.misses + other.misses,
            self.false_alarms + other.false_alarms,
            self.correct_rejections + other.correct_rejections
        )

    def __sub__(self, other):
        return SignalDetection(
            self.hits - other.hits,  # need to ccount for negative cases
            self.misses - other.misses,
            self.false_alarms - other.false_alarms,
            self.correct_rejections - other.correct_rejections
        )
    
    def __mul__(self, factor):
        return SignalDetection(
            self.hits * factor,  # need to ccount for negative cases
            self.misses * factor ,
            self.false_alarms * factor,
            self.correct_rejections * factor
        )

    def plot_sdt(self):
        fig, ax = plt.subplots()

        x = np.linspace(-8, 8, 1000)
        noise_dist = norm.pdf(x, loc=0, scale=1)
        signal_dist = norm.pdf(x, loc=self.d_prime(), scale=1)

        ax.plot(x, noise_dist, label="noise")
        ax.plot(x, signal_dist, label="signal")
        ax.axvline(self.criterion(), linestyle='--', color='r')
        # ax.axvline(0, linestyle='--')
        # ax.axvline(self.d_prime(), linestyle='--')
        ax.annotate(
            '', 
            xy=(self.d_prime(), 0.1), 
            xytext=(0, 0.1),
            arrowprops=dict(arrowstyle='<->')
        )

        ax.set_ylabel("Density")
        ax.set_title("SDT plot")
        ax.legend()

        return fig, ax


def main():
    test = SignalDetection(180, 30, 40, 160)
    print("Hite rate:", test.hit_rate())
    print("False alarm rate:", test.false_alarm_rate())
    print("Sensitivity:", test.d_prime())
    print("Criterion:", test.criterion())
    fig, ax = test.plot_sdt()
    fig.savefig("sdt_plot_2.png")


if __name__ == "__main__":
    main()



        
    

    
