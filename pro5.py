import numpy as np
import matplotlib.pyplot as plt

def chebyshev_diff_matrix(N):
    if N == 0:
        return np.array([[0.]]), np.array([0.])
        
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = c[N] = 2.
    
    D = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            if i == j:
                if i == 0:
                    D[i, j] = (2 * N**2 + 1) / 6.
                elif i == N:
                    D[i, j] = -(2 * N**2 + 1) / 6.
                else:
                    D[i, j] = -x[j] / (2 * (1 - x[j]**2))
            else:
                D[i, j] = (c[i] / c[j]) * ((-1)**(i + j)) / (x[i] - x[j])
                
    return D, x

# problem
u_exact_func = lambda x: np.cos(np.pi * x) + 1.0
g_func = lambda x: (np.pi**4 + 4*np.pi**2 + 3) * np.cos(np.pi * x) + 3.0

N_values = [8, 16, 32, 64]

plt.style.use('seaborn-v0_8-whitegrid')
fig_sol, ax_sol = plt.subplots(figsize=(10, 6))

# plot exact solution
x_fine = np.linspace(-1, 1, 500)
ax_sol.plot(x_fine, u_exact_func(x_fine), 'k-', linewidth=2, label='Exact Solution')

for N in N_values:
    # get Chebyshev differentiation matrix and nodes
    D, x = chebyshev_diff_matrix(N)
    D2 = np.dot(D, D)
    D4 = np.dot(D2, D2)
    L = D4 - 4 * D2 + 3 * np.identity(N + 1)
    g = g_func(x)
    
    # reduce to interior points
    L_interior = L[1:-1, 1:-1]
    g_interior = g[1:-1]
    
    # impose derivative BCs
    L_interior[0, :] = D[0, 1:-1]   # u'(1) = 0
    g_interior[0] = 0
    L_interior[-1, :] = D[N, 1:-1]  # u'(-1) = 0
    g_interior[-1] = 0

    # solve for the interior points
    u_interior = np.linalg.solve(L_interior, g_interior)

    # add boundary values
    u_numerical = np.concatenate(([0], u_interior, [0]))
    
    ax_sol.plot(x, u_numerical, 'o-', markersize=6, label=f'Numerical N = {N}')

ax_sol.set_title('Exact vs Numerical solutions', fontsize=16)
ax_sol.set_xlabel('x', fontsize=12)
ax_sol.set_ylabel('u(x)', fontsize=12)
ax_sol.legend()
ax_sol.grid(True)
fig_sol.savefig('solution_comparison.png', dpi=300)


n_test_range = range(8, 129)
errors_for_plot = []
n_for_plot = []
n_accuracy_limit = -1

for n_test in n_test_range:
    D, x = chebyshev_diff_matrix(n_test)
    D2 = np.dot(D, D)
    D4 = np.dot(D2, D2)
    L = D4 - 4 * D2 + 3 * np.identity(n_test + 1)
    g = g_func(x)

    L_interior = L[1:-1, 1:-1]
    g_interior = g[1:-1]
    L_interior[0, :] = D[0, 1:-1]
    g_interior[0] = 0
    L_interior[-1, :] = D[n_test, 1:-1]
    g_interior[-1] = 0
    
    u_interior = np.linalg.solve(L_interior, g_interior)
    u_numerical = np.concatenate(([0], u_interior, [0]))
    max_error = np.max(np.abs(u_numerical - u_exact_func(x)))
    
    errors_for_plot.append(max_error)
    n_for_plot.append(n_test)
    
    print(f"Testing N = {n_test}, Max Error = {max_error:.2e}")


fig_conv, ax_conv = plt.subplots(figsize=(10, 6))
ax_conv.semilogy(n_for_plot, errors_for_plot, 'o-', markersize=6, label='Max Error')
ax_conv.set_title('Maximum error vs N', fontsize=16)
ax_conv.set_xlabel('N', fontsize=12)
ax_conv.set_ylabel('Max error', fontsize=12)
ax_conv.grid(True, which='both', linestyle='--')
fig_conv.savefig('convergence_plot.png', dpi=300)

plt.show()
