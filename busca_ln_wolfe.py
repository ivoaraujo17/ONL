import numpy as np
from scipy.optimize import approx_fprime as gradient


# busca Wolfe
def busca_linear_wolfe(f, x, d, t, y=0.8, n1=1e-2, n2=1e-1):
    k=0
    fx = f(x)
    g_fx = gradient(x, f, epsilon=1e-5) # gradiente numérico
    fxtd = f(x + t*d)

    print(f'\n----------------------------------------iteração = {k} -------------------------------------------\n')
    print('lado esquerdo')
    print("f(    x    +    t    *    d    ) <= f(x)        +    n1   *   t    * grad(f(     x     )) *    d    ")
    print("f({:^9}+{:^9}*{:^9}) <= f({:^8}) + {:^7} * {:^5}  * grad(f({:^11})) * {:^8}"
        .format(str(x),str(t),str(d),str(x),str(n1),str(t),str(x),str(d)))
    print("f({:^9}+{:^19}) <= {:^11} + {:^7} * {:^5}  * {:^20} * {:^8}"
        .format(str(x),str(t*d),str(fx),str(n1),str(t),str(g_fx),str(d)))
    print("f({:^29}) <= {:^11} + {:^7} * {:^5}  * {:^31}"
        .format(str(x + t*d), str(fx), str(n1),str(t),str(np.dot(g_fx, d))))
    print("{:^32} <= {:<63}"
        .format(str(fxtd), str(fx + n1*t*np.dot(g_fx, d)),))
    
    print('\nlado direito')
    print("grad(f(    x    +    t    *    d    )) *    d     >=    n2    * grad(f(     x     )) *    d    ")
    print("grad(f({:^9}+{:^9}*{:^9})) * {:^8} >= {:^7}  * grad(f({:^11})) * {:^8}"
        .format(str(x),str(t),str(d), str(d),str(n2),str(x),str(d)))
    print("grad(f({:^29})) * {:^8} >= {:^7}  * {:^20} * {:^8}"
        .format(str(x + t*d),str(d),str(n2),str(g_fx),str(d)))
    print("{:>49} >= {:<63}"
        .format(str(np.dot(gradient(x+t*d, f, epsilon=1e-5), d)), str(n2*np.dot(g_fx, d)),))
    
    while fxtd > fx + n1*t*np.dot(g_fx, d) \
            or \
            np.dot(gradient(x+t*d, f, epsilon=1e-5), d) <= n2*np.dot(g_fx, d):
        t *= y
        t = round(t, 5)
        k += 1
        fxtd = f(x + t*d)

        print(f'\n----------------------------------------iteração = {k} -------------------------------------------\n')
        print('lado esquerdo')
        print("f(    x    +    t    *    d    ) <= f(x)        +    n1   *   t    * grad(f(     x     )) *    d    ")
        print("f({:^9}+{:^9}*{:^9}) <= f({:^8}) + {:^7} * {:^5}  * grad(f({:^11})) * {:^8}"
            .format(str(x),str(t),str(d),str(x),str(n1),str(t),str(x),str(d)))
        print("f({:^9}+{:^19}) <= {:^11} + {:^7} * {:^5}  * {:^20} * {:^8}"
            .format(str(x),str(t*d),str(fx),str(n1),str(t),str(g_fx),str(d)))
        print("f({:^29}) <= {:^11} + {:^7} * {:^5}  * {:^31}"
            .format(str(x + t*d), str(fx), str(n1),str(t),str(np.dot(g_fx, d))))
        print("{:^32} <= {:<63}"
            .format(str(fxtd), str(fx + n1*t*np.dot(g_fx, d)),))
        
        print('\nlado direito')
        print("grad(f(    x    +    t    *    d    )) *    d     >=    n2    * grad(f(     x     )) *    d    ")
        print("grad(f({:^9}+{:^9}*{:^9})) * {:^8} >= {:^7}  * grad(f({:^11})) * {:^8}"
            .format(str(x),str(t),str(d), str(d),str(n2),str(x),str(d)))
        print("grad(f({:^29})) * {:^8} >= {:^7}  * {:^20} * {:^8}"
            .format(str(x + t*d),str(d),str(n2),str(g_fx),str(d)))
        print("{:>49} >= {:<63}"
            .format(str(np.dot(gradient(x+t*d, f, epsilon=1e-5), d)), str(n2*np.dot(g_fx, d)),))

        if t < 1e-8:
            print('Erro no Backtracking')
            return [False, t, k]
    return [True, t, k]