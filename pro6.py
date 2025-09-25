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
s_to_x = lambda s: 2.5 * s + 2.5
x_to_s = lambda x: 0.4 * x - 1.0

# exact solution and g(x)
u_exact_func = lambda x: 1.0 / (1.0 + x**2)
g_func = lambda x: (24 - 240*x**2 + 120*x**4) / (1 + x**2)**5 + 1 / (1 + x**2) - 2*x / (1 + x**2)**2

# boundary conditions
U0, U5 = 1.0, 1.0/26.0
U0_prime_x, U5_prime_x = 0.0, -10.0/676.0 # Derivatives wrt x

U0_prime_s = U0_prime_x * 2.5
U5_prime_s = U5_prime_x * 2.5

# homogenization function b(s)
p0 = np.poly1d([1, 0, -3, 2]) / 4
q0 = np.poly1d([1, -1, -1, 1]) / 4
p1 = np.poly1d([-1, 0, 3, 2]) / 4
q1 = np.poly1d([1, 1, -1, -1]) / 4

b_poly = p0 * U0 + q0 * U0_prime_s + p1 * U5 + q1 * U5_prime_s

b_prime_poly = b_poly.deriv(1)
b_s = lambda s: b_poly(s)
b_prime_s = lambda s: b_prime_poly(s)

def solve_bvp_p6(N):
    D, s = chebyshev_diff_matrix(N)
    x = s_to_x(s)
    
    I = np.identity(N + 1)
    D2 = np.dot(D, D)
    D4 = np.dot(D2, D2)
    
    L_v = (2/5)**4 * D4 + (2/5) * D + I
    
    rhs_g = g_func(x)
    rhs_b = (2/5) * b_prime_s(s) + b_s(s)
    rhs = rhs_g - rhs_b
    
    # reduce to interior points
    L_v_int = L_v[1:-1, 1:-1]
    rhs_int = rhs[1:-1]
    
    L_v_int[0, :] = D[0, 1:-1]
    rhs_int[0] = 0
    L_v_int[-1, :] = D[N, 1:-1]
    rhs_int[-1] = 0
    
    # solve for interior points
    v_int = np.linalg.solve(L_v_int, rhs_int)
    v = np.concatenate(([0], v_int, [0]))
    
    u = v + b_s(s)
    
    return x, u

N_values_plot = [8, 16, 32, 64]
plt.style.use('seaborn-v0_8-whitegrid')
fig_sol, ax_sol = plt.subplots(figsize=(10, 7))

x_fine = np.linspace(0, 5, 500)
ax_sol.plot(x_fine, u_exact_func(x_fine), 'k-', linewidth=2, label='Exact Solution')

for N in N_values_plot:
    x, u = solve_bvp_p6(N)
    ax_sol.plot(x, u, 'o-', markersize=6, label=f'Numerical N = {N}')

ax_sol.set_title('Exact vs Numerical solutions', fontsize=16)
ax_sol.set_xlabel('x', fontsize=12)
ax_sol.set_ylabel('u(x)', fontsize=12)
ax_sol.legend()
ax_sol.grid(True)
fig_sol.savefig('p6_solution_comparison.png', dpi=300)

n_test_range = range(8, 129)
errors = []
n_vals = []
n_machine_precision = -1

for N in n_test_range:
    x, u = solve_bvp_p6(N)
    error = u - u_exact_func(x)
    max_error = np.max(np.abs(error))
    errors.append(max_error)
    n_vals.append(N)
    print(f"Testing N = {N}, Max Error = {max_error:.2e}")

fig_err, ax_err = plt.subplots(figsize=(10, 7))
ax_err.semilogy(n_vals, errors, 'o-')
ax_err.set_title('Maximum error vs N', fontsize=16)
ax_err.set_xlabel('N', fontsize=12)
ax_err.set_ylabel('Max error', fontsize=12)
ax_err.grid(True, which='both')
fig_err.savefig('p6_convergence.png', dpi=300)

plt.show()