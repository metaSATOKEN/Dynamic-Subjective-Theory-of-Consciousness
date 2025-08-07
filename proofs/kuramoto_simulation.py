"""
Simulate N Kuramoto oscillators and compute strength synchronization λ(t).

Verification target:
  λ(t) = |mean(exp(iθ))|²
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 100
T = 1000
dt = 0.05
K = 1.5
omega = np.random.normal(0, 1, N)
theta = np.random.uniform(0, 2*np.pi, N)
lambda_t = []

# Simulation loop
for _ in range(T):
    theta += (omega + K/N * np.sum(np.sin(theta[:,None] - theta), axis=1)) * dt
    r = np.abs(np.mean(np.exp(1j * theta)))
    lambda_t.append(r**2)

# Plot
t = np.arange(T) * dt
plt.plot(t, lambda_t)
plt.xlabel('Time')
plt.ylabel('λ(t)')
plt.title('Strength Synchronization over Time')
plt.grid()
plt.show()
