import numpy as np

def calcular_estatisticas_descritivas(dados):
    """
    Calcula e retorna estatísticas descritivas básicas de um conjunto de dados.

    Args:
        dados (array-like): Um array ou lista de números.

    Returns:
        dict: Um dicionário contendo n, média, desvio padrão e variância.
    """
    dados = np.asarray(dados)
    media = np.mean(dados)
    desvio_padrao = np.std(dados, ddof=1)  # ddof=1 para desvio padrão amostral
    variancia = np.var(dados, ddof=1)    # ddof=1 para variância amostral
    n = len(dados)
    return {
        'n': n,
        'media': media,
        'desvio_padrao': desvio_padrao,
        'variancia': variancia
    }

def ordenar_dados(dados):
    """
    Retorna os dados ordenados em ordem crescente.

    Args:
        dados (array-like): Um array ou lista de números.

    Returns:
        np.ndarray: Um array numpy com os dados ordenados.
    """
    return np.sort(np.asarray(dados))

def padronizar_dados(dados):
    """
    Padroniza os dados para terem média 0 e desvio padrão 1 (Z-score).

    Args:
        dados (array-like): Um array ou lista de números.

    Returns:
        np.ndarray: Um array numpy com os dados padronizados.
    """
    dados = np.asarray(dados)
    stats = calcular_estatisticas_descritivas(dados)
    media = stats['media']
    desvio_padrao = stats['desvio_padrao']

    if desvio_padrao == 0:
        # Evita divisão por zero se todos os dados forem iguais
        return np.zeros_like(dados)

    return (dados - media) / desvio_padrao


from scipy.stats import norm

def lilliefors_critico(n, alpha=0.05, n_sim=100000):
    """
    Calcula e retorna o valor crítico para o teste de Lilliefors.
    Args:
        n (int): tamanho de amostra
        alpha (float): float (nível de significância)
                       Probabilidade de erro tipo I. Exemplo: alpha=0.05 significa 5% de chance
                       de rejeitar a hipótese de normalidade quando ela é verdadeira.
    m_sim
    """
    D_vals = []
    for _ in range(n_sim):
        sample = np.random.normal(0, 1, n)
        mu, sigma = np.mean(sample), np.std(sample, ddof=0)
        z = padronizar_dados(sample)
        z.sort()
        F0 = norm.cdf(z)
        i = np.arange(1, n+1)
        D = np.max(np.maximum(i/n - F0, F0 - (i-1)/n))
        D_vals.append(D)
    return np.quantile(D_vals, 1 - alpha)

def cdf(z):
    return norm.cdf(z)

def mean(x):
    return np.mean(x)
