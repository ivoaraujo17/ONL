import numpy as np
from scipy.optimize import approx_fprime as gradient


# busca Goldstein
def busca_linear_goldstein(f, x, d, t, y=0.5, n=1e-4):

    print("\n------Busca Linear Goldstein------\n")
    print("parametros:")
    print("f = ", f)
    print("x = ", x)
    print("d = ", d)
    print("t = ", t)
    print("y = ", y)
    print("n = ", n)
    k=0
    fx = f(x)
    fxtd = f(x + t*d)
    g_fx = gradient(x, f, epsilon=1e-10) # gradiente

    print(f'\n----------------------------------------iteração = {k} -----------------------------------------------\n')
    print('lado esquerdo')
    print("f(    x   ) + (1 -    n  ) *   t   * grad(f(     x     )) *     d    <= f(    x    +    t    *    d    )")
    print("f({:^8}) + ({} - {:^6}) * {:^5} * grad(f({:^11})) * {:^8} <= f({:^9}+{:^9}*{:^9})"
        .format(str(x),str(1),str(n),str(t),str(x),str(d), str(x),str(t),str(d)))
    print("{:^11} + ({} - {:^6}) * {:^5} * {:^20} * {:^8} <= f({:^9}+{:^19})"
        .format(str(fx),str(1),str(n),str(t),str(g_fx),str(d), str(x),str(t*d)))
    print("{:^11} + ({} - {:^6}) * {:^5} * {:^31} <= f({:^29})"
        .format(str(fx),str(1),str(n),str(t),str(np.dot(g_fx, d)), str(x + t*d)))
    print("{:^68} <= {:^29}"
            .format(str(fx + (1-n)*t*np.dot(g_fx, d)),fxtd))
    
    print('\nlado direito')
    print("f(    x    +    t    *    d    ) <= f(x)        +    n    *   t   * grad(f(     x     )) *    d    ")
    print("f({:^9}+{:^9}*{:^9}) <= f({:^8}) + {:^7} * {:^5} * grad(f({:^11})) * {:^8}"
        .format(str(x),t,str(d),str(x),str(n),str(t),str(x),str(d)))
    print("f({:^9}+{:^19}) <= {:^11} + {:^7} * {:^5} * {:^20} * {:^8}"
        .format(str(x),str(t*d),str(fx),str(n),str(t),str(g_fx),str(d)))
    print("f({:^29}) <= {:^11} + {:^7} * {:^5} * {:^31}"
        .format(str(x + t*d), str(fx), str(n),str(t),str(np.dot(g_fx, d))))
    print("{:^32} <= {:<63}"
        .format(str(fxtd), str(fx + n*t*np.dot(g_fx, d)),))
    while fxtd > fx + n*t*np.dot(g_fx, d) \
            or fx + (1-n)*t*np.dot(g_fx, d) > fxtd:

        t *= y
        t = round(t, 5)
        fxtd = f(x + t*d)
        k += 1

        print(f'\n----------------------------------------iteração = {k} -----------------------------------------------\n')
        print('lado esquerdo')
        print("f(    x    +    t    *    d    ) <= f(x)        +    n    *   t   * grad(f(     x     )) *    d    ")
        print("f({:^8}) + ({} - {:^6}) * {:^5} * grad(f({:^11})) * {:^8} <= f({:^9}+{:^9}*{:^9})"
        .format(str(x),str(1),str(n),str(t),str(x),str(d), str(x),str(t),str(d)))
        print("{:^11} + ({} - {:^6}) * {:^5} * {:^20} * {:^8} <= f({:^9}+{:^19})"
            .format(str(fx),str(1),str(n),str(t),str(g_fx),str(d), str(x),str(t*d)))
        print("{:^11} + ({} - {:^6}) * {:^5} * {:^31} <= f({:^29})"
            .format(str(fx),str(1),str(n),str(t),str(np.dot(g_fx, d)), str(x + t*d)))
        print("{:^68} <= {:^29}"
                .format(str(fx + (1-n)*t*np.dot(g_fx, d)),fxtd))

        print('\nlado direito')
        print("f(    x    +    t    *    d    ) <= f(x)        +    n    *   t   * grad(f(     x     )) *    d    ")
        print("f({:^9}+{:^9}*{:^9}) <= f({:^8}) + {:^7} * {:^5} * grad(f({:^11})) * {:^8}"
            .format(str(x),t,str(d),str(x),str(n),str(t),str(x),str(d)))
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