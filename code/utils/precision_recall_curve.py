import matplotlib.pyplot as plt
from utils.tools import draw_range
# Precision Recall Curve data

markers = "DdsPvo*xH1234h"
method2marker = {}
i = 0
for method in pr_data:
    method2marker[method] = markers[i]
    i += 1

plt.figure(figsize=(20, 5))
plt.subplot(131)
for method in pr_data:
    P, R = pr_data[method]
    plt.plot(R, P, linestyle="-", marker=method2marker[method], label=method)
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend()

plt.subplot(132)
for method in pr_data:
    P, R = pr_data[method]
    plt.plot(draw_range, R, linestyle="-", marker=method2marker[method], label=method)
plt.xlim(0, max(draw_range))
plt.grid(True)
plt.xlabel('The number of retrieved samples')
plt.ylabel('recall')
plt.legend()

plt.subplot(133)
for method in pr_data:
    P, R = pr_data[method]
    plt.plot(draw_range, P, linestyle="-", marker=method2marker[method], label=method)
plt.xlim(0, max(draw_range))
plt.grid(True)
plt.xlabel('The number of retrieved samples')
plt.ylabel('precision')
plt.legend()
plt.savefig("pr.png")
plt.show()
