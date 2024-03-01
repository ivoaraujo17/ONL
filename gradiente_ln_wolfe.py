from busca_ln_wolfe import busca_linear_wolfe
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import approx_fprime as gradient


# metodo do gradiente com wolfe
def gradiente_ln_wolfe(f, x):
    g = gradient(x, f, epsilon=1e-5)
    n1 = 1e-2
    n2 = 1e-1
    y = 0.5 # fator de redução de t
    num_backtrack = 0
    iter = 0

    # Definindo limites do gráfico
    x_min, x_max = -20, 20
    y_min, y_max = -20, 20
    step = 0.1
    X, Y = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    Z = f([X, Y])
    
    plt.contour(X, Y, Z, levels=np.logspace(0, 3, 10))
    plt.xlabel('x')
    plt.ylabel('y')
    # Plotando o ponto inicial
    plt.plot(x[0], x[1], 'ro')

    # Lista para armazenar os pontos iterativos
    pontos_x = [x]

    while np.linalg.norm(g) > 1e-10:
        d = -g
        t = 1
        result, t, k = busca_linear_wolfe(f, x, d, t, y, n1, n2)
        if not result:
            print("Não Converge")
            return -1
        xt= x + t * d
        x=xt

        # Adicionando o novo ponto à lista
        pontos_x.append(x)

        g = gradient(x, f, epsilon=1e-5)
        iter += 1
        num_backtrack += k
        if iter > 100000:
            print("Não Converge")
            return -1
    
    # Convertendo a lista de pontos em um array numpy para facilitar o acesso
    pontos_x = np.array(pontos_x)

    # Plotando a linha conectando os pontos iterativos
    plt.plot(pontos_x[:, 0], pontos_x[:, 1], 'bo-')

    plt.show()

    return x, iter, num_backtrack
