import numpy as np
import matplotlib.pyplot as plt

std_iniziali_frac = np.array([0.00255, 0.00325, 0.00425, 0.00525, 0.00625, 0.00725, 0.00825, 0.00925, 0.01025, 0.01125, 0.01225])

std_finali_frac = np.array([3.8295, 3.5208, 3.4213, 3.4336, 3.4771, 3.3928, 3.5502, 3.2795, 3.7931, 4.0263, 3.8753])


plt.figure(figsize=(8, 6))
plt.plot(std_iniziali_frac, std_finali_frac, marker='o', linestyle='-', color='b')
plt.title('Final Standard Deviation vs Initial Standard Deviation')
plt.xlabel('Initial Standard Deviation')
plt.ylabel('Final Standard Deviation')
plt.grid()
# Save the figure
plt.savefig('std_final_vs_initial.png', dpi=300)
plt.show()