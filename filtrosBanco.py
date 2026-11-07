
import numpy as np

# ---------------- PARAMETROS (ajustables) ----------------
FS = 8000                 # frecuencia de muestreo (Hz)
N_ORDEN = 255             # orden del FIR (impar, como en las notas)
VENTANA = "hamming"       # "hamming" o "hanning"
BANDAS = [                # entre 2 y 5 subbandas (fc1, fc2) en Hz
    (80, 250),
    (250, 800),
    (800, 2000),
    (2000, 3800),
]
DUR_TRAMA = 0.02          # 20 ms por trama para calcular energias
DUR_GRABACION = 2.0       # segundos de captura del comando de voz

def omega_c(fc, fs):
    return 2 * np.pi * fc / fs


def h_pasabanda_ideal(fc1, fc2, fs, N):

    w1, w2 = omega_c(fc1, fs), omega_c(fc2, fs)
    M = (N - 1) // 2
    n = np.arange(-M, M + 1)
    h = np.zeros(N)
    nz = n != 0
    h[nz] = (np.sin(w2 * n[nz]) - np.sin(w1 * n[nz])) / (np.pi * n[nz])
    h[~nz] = (w2 - w1) / np.pi
    return h


def ventana(N, tipo="hamming"):
    n = np.arange(N)
    if tipo == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    if tipo == "hanning":
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    raise ValueError("Ventana no soportada")


def disenar_banco(fs=FS, N=N_ORDEN, bandas=BANDAS, tipo=VENTANA):
    assert 2 <= len(bandas) <= 5, "El enunciado pide entre 2 y 5 subbandas"
    assert N % 2 == 1, "N debe ser impar"
    return [h_pasabanda_ideal(f1, f2, fs, N) * ventana(N, tipo) for f1, f2 in bandas]


def filtrar(x, hd):
    return np.convolve(x, hd, mode="same")


def energias_por_trama(y, fs=FS, dur=DUR_TRAMA):
    L = int(dur * fs)
    T = len(y) // L
    tramas = y[: T * L].reshape(T, L)
    return np.sum(tramas ** 2, axis=1)


def caracteristicas(x, banco, fs=FS):

    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    x = x / (np.max(np.abs(x)) + 1e-12)          # normalizacion de amplitud

    E = np.array([energias_por_trama(filtrar(x, hd), fs) for hd in banco])  # (K, T)

    # Descartar tramas de silencio (energia total baja)
    total = E.sum(axis=0)
    activa = total > 0.05 * total.max()
    E = E[:, activa]

    e_prom = E.mean(axis=1)
    e_desv = E.std(axis=1)
    return np.concatenate([e_prom, e_desv])