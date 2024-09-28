from scipy import optimize
from multiprocessing import Pool, cpu_count
import math
import mpmath
import time
import sympy
import cmath

mu = [1, 1, -1, -1, 0, -1, 1, -1]

def nEMB(t, d):
    a = 0.451
    b = 1.407 * math.sqrt(d) - 0.245
    c = 0.371 * d + 0.195
    return math.ceil(a * t + b * math.sqrt(t) + c)

def cMB(n):
    K = math.floor(n / math.sqrt(2))
    T = [None] * (n + 1)
    T[0] = -math.log(n)
    for k in range(1, n + 1):
        T[k] = T[k - 1] + math.log(n - k + 1) + math.log(n + k - 1) - math.log(2 * k - 1) - math.log(2 * k)
    SR = [None] * (n + 1)
    SR[0] = math.exp(T[0] - T[K] - K * math.log(4))
    for k in range(1, n + 1):
        SR[k] = SR[k - 1] + math.exp(T[k] - T[K] + math.log(4) * (k - K))
    cnk = [None] * (n)
    for k in range(0, n):
        cnk[k] = (1 - SR[k] / SR[n])
    return cnk

def ZETAkuzmas(s, c):
    S = complex(0, 0)
    p = 1
    for k, val in enumerate(c):
        S += (p * c[k] / pow(k + 1, s))
        p = -p
    return S / (1.0 - pow(2.0, -s + 1.0))

def ZETAX(s, d):
    n = nEMB(s.imag, d)
    c = cMB(n)
    z = ZETAkuzmas(s, c)
    return z

def Prime_Zeta_CohenX(sigma, t):
    MP = 10 # buvo 17
    Reciprocal_P = [0 for i in range(MP)]
    for k in range(MP):
        Reciprocal_P[k] = 1 / sympy.prime(k + 1)
    argumentas = complex(sigma, t)
    S1 = 0
    for k in range(MP):
        S1 += Reciprocal_P[k] ** argumentas
    S2 = 0
    for k in range(1, 5): # buvo 40
        P_sandauga = 1
        for i in range(MP):
            P_sandauga = P_sandauga * (1 - Reciprocal_P[i] ** (argumentas * k))
        S2 += (mu[k] / k) * cmath.log(ZETAX(k * argumentas, 8) * P_sandauga)
    return S1 + S2

def objective_function(x): return abs(Prime_Zeta_CohenX(x[0], x[1])) ** 2

def run_parallel(n):
  text ="Total cores: " + str(cpu_count()) + ", used cores: " + str(n)
  print(text)
  pool = Pool(processes = n)
  with open("output_CohenX_gap1.txt", "w") as file:
    file.write("%s\n" % text)
    print("t1 / t2 / sigma / t / obj_value / calc_time")
    file.write("%s\t%s\t%s\t%s\t%s\t%s\t\n" % ("sigma1", "sigma2", "sigma", "t", "obj_value", "calc_time"))
    for i in range(10000):
      start = time.time()
      bounds = [(sigma1, sigma2), (gap * i, gap * (i + 1))]
      result = optimize.differential_evolution(objective_function, bounds, updating='deferred', workers = n)
      end = time.time()
      file.write("%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.3f\n" % (gap * i, gap * (i + 1), result.x[0], result.x[1], result.fun, end - start))
      file.flush()
      print(gap * i, gap * (i + 1), result.x[0], result.x[1], result.fun, end - start)

if __name__ == '__main__':
  sigma1 = 1 # sigma1 = 1.57175 # old
  sigma2 = 1.78 # sigma2 = 1.77955 # old
  gap = 1 # gap sequence
  
  run_parallel(31) # number of used CPU cores for computation