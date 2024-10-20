import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def q2():
    mean_diffs = [2, 4, 1, 0, 7, 5]
    print(np.array(sorted(range(len(mean_diffs)), key=lambda i: mean_diffs[i], reverse=True)))

    set0 = [[1,2], [11,12]]

    set0_avg = np.asarray(set0).mean(axis=0)
    print(set0_avg)

    set0_avg = np.asarray(set0).mean(axis=1)
    print(set0_avg)

    def hamw(v):
        c = 0
        while v:
            c += 1
            v &= v - 1

        return c

    print(hamw(0x00))





n = norm(0,.5)

x = np.linspace(-3, 3, 1000)

pdf = n.pdf(x)

# Plot the PDF
plt.plot(x, pdf)
plt.title('Normal Distribution (mean=0, std=0.5)')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
