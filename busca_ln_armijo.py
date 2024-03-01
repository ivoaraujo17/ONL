import numpy as np
from scipy.optimize import approx_fprime as gradient


# busca armijo
def busca_linear_armijo(f, x, d, t, y=0.8, n=0.25):
    k=0
    fx = f(x)
    g_fx = gradient(x, f, epsilon=1e-5) # gradiente
    fxtd = f(x + t*d)

    print(f'\n----------------------------------------iteração = {k} -------------------------------------------\n')
    print("f(    x    +    t    *    d    ) <= f(x)        +    n    *   t   * grad(f(     x     )) *    d    ")
    print("f({:^9}+{:^9}*{:^9}) <= f({:^8}) + {:^7} * {:^5} * grad(f({:^11})) * {:^8}"
        .format(str(x),str(t),str(d),str(x),str(n),str(t),str(x),str(d)))
    print("f({:^9}+{:^19}) <= {:^11} + {:^7} * {:^5} * {:^20} * {:^8}"
        .format(str(x),str(t*d),str(fx),str(n),str(t),str(g_fx),str(d)))
    print("f({:^29}) <= {:^11} + {:^7} * {:^5} * {:^31}"
        .format(str(x + t*d), str(fx), str(n),str(t),str(np.dot(g_fx, d))))
    print("{:^32} <= {:<63}"
        .format(str(fxtd), str(fx + n*t*np.dot(g_fx, d)),))

    while fxtd > fx + n*t*np.dot(g_fx, d):
        t *= y
        t = round(t, 5)
        k += 1
        fxtd = f(x + t*d)

        print(f'\n----------------------------------------iteração = {k} -------------------------------------------\n')
        print("f(    x    +    t    *    d    ) <= f(x)        +    n    *   t   * grad(f(     x     )) *    d    ")
        print("f({:^9}+{:^9}*{:^9}) <= f({:^8}) + {:^7} * {:^5} * grad(f({:^11})) * {:^8}"
            .format(str(x),str(t),str(d),str(x),str(n),str(t),str(x),str(d)))
        print("f({:^9}+{:^19}) <= {:^11} + {:^7} * {:^5} * {:^20} * {:^8}"
            .format(str(x),str(t*d),str(fx),str(n),str(t),str(g_fx),str(d)))
        print("f({:^29}) <= {:^11} + {:^7} * {:^5} * {:^31}"
            .format(str(x + t*d), str(fx), str(n),str(t),str(np.dot(g_fx, d))))
        print("{:^32} <= {:<63}"
            .format(str(fxtd), str(fx + n*t*np.dot(g_fx, d)),))

        if t < 1e-8:
            print('Erro no Backtracking')
            return [False, t, k]
    return [True, t, k]