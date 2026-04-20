from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from numbers import Number

class SignalDetection:
    def __init__(self, hits, misses, false_alarms, correct_rejections):
        if any(isinstance(v, Number) == False for v in (hits, misses, false_alarms, correct_rejections)):
            raise TypeError("Not numeric value(s)")
        if any(v < 0 for v in (hits, misses, false_alarms, correct_rejections)):
            raise ValueError("Negative values(s)")

        self.hits = hits
        self.misses = misses
        self.false_alarms = false_alarms
        self.correct_rejections = correct_rejections

    def hit_rate(self):
        total_1 = self.hits + self.misses
        if total_1 == 0:
            raise ValueError("0 for hits + misses")
        return self.hits / total_1

    def false_alarm_rate(self):
        total_0 = self.false_alarms + self.correct_rejections
        if total_0 == 0:
            raise ValueError("0 for false alarms + correct_rejections")
        return self.false_alarms / total_0
    
    def d_prime(self):
        hit_rate = self.hit_rate()
        false_alarm_rate = self.false_alarm_rate()

        if hit_rate == 0:
            hit_rate += 1e-12
        elif hit_rate == 1:
            hit_rate -= 1e-12
        if false_alarm_rate == 0:
            false_alarm_rate += 1e-12
        elif false_alarm_rate == 1:
            false_alarm_rate -= 1e-12

        return norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)
    
    def criterion(self):
        hit_rate = self.hit_rate()
        false_alarm_rate = self.false_alarm_rate()

        if hit_rate == 0:
            hit_rate += 1e-12
        elif hit_rate == 1:
            hit_rate -= 1e-12
        if false_alarm_rate == 0:
            false_alarm_rate += 1e-12
        elif false_alarm_rate == 1:
            false_alarm_rate -= 1e-12

        return -0.5 * (norm.ppf(hit_rate) + norm.ppf(false_alarm_rate))
    
    def __add__(self, other):
        if not isinstance(other, SignalDetection):
            raise TypeError("Invalid object type")

        return SignalDetection(
            self.hits + other.hits,
            self.misses + other.misses,
            self.false_alarms + other.false_alarms,
            self.correct_rejections + other.correct_rejections
        )

    def __sub__(self, other):
        if not isinstance(other, SignalDetection):
            raise TypeError("Invalid object type")

        new_hits = self.hits - other.hits
        new_misses = self.misses - other.misses
        new_false_alarms = self.false_alarms - other.false_alarms
        new_correct_rejections = self.correct_rejections - other.correct_rejections
        
        if any(v < 0 for v in (new_hits, new_misses, new_false_alarms, new_correct_rejections)):
            raise ValueError("Negative values(s) after subtraction") 
        return SignalDetection(new_hits, new_misses, new_false_alarms, new_correct_rejections)
    
    def __mul__(self, factor):
        scaled_hits = self.hits * factor
        scaled_misses = self.misses * factor
        scaled_false_alarms = self.false_alarms * factor
        scaled_correct_rejections = self.correct_rejections * factor

        if any(v < 0 for v in (scaled_hits, scaled_misses, scaled_false_alarms, scaled_correct_rejections)):
            raise ValueError("Negative values(s) after scalar multiplication") 
        return SignalDetection(scaled_hits, scaled_misses, scaled_false_alarms, scaled_correct_rejections)


    def plot_sdt(self):
        fig, ax = plt.subplots()
        x = np.linspace(-8, 8, 1000)
        noise_dist = norm.pdf(x, loc=0, scale=1)
        signal_dist = norm.pdf(x, loc=self.d_prime(), scale=1)

        ax.plot(x, noise_dist, label="noise", color="blue")
        # ax.axvline(0, linestyle='--', color="blue")
        ax.plot(x, signal_dist, label="signal", color="orange")
        # ax.axvline(self.d_prime(), linestyle='--', color="orange")

        ax.axvline(self.criterion(), color='grey', linestyle='--')
        ax.text(self.criterion()+0.1, 0.02, 'criterion', color='grey')
        ax.annotate(
            '', 
            xy=(self.d_prime(), 0.4), 
            xytext=(0, 0.4),
            arrowprops=dict(arrowstyle='<->')
        )
        ax.text(self.d_prime()/2, 0.405, "d'", ha='center')
        ax.set_ylim(-0.02, 0.43)
        ax.set_ylabel("Density")
        ax.set_title("SDT plot")
        ax.legend()

        return fig, ax


def main():
    # test = SignalDetection(0, 1, 1, 1) # invalid input test
    test = SignalDetection(100, 50, 30, 120)
    print("Hite rate:", test.hit_rate())
    print("False alarm rate:", test.false_alarm_rate())
    print("Sensitivity:", test.d_prime())
    print("Criterion:", test.criterion())
    fig, ax = test.plot_sdt()
    fig.savefig("sdt_plot.png")


if __name__ == "__main__":
    main()



        
    

    
