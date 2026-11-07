
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# ---------------- PARAMETROS (ajustables) ----------------
FS = 8000                 # frecuencia de muestreo (Hz)
N_ORDEN = 255             # orden del FIR (impar, como en las notas)
VENTANA = "hamming"       # "hamming" o "hanning"
BANDAS = [                # entre 2 y 5 subbandas (fc1, fc2) en Hz
    (85, 180),
    (180, 255),
    (255, 1000),
    (1000, 3800),
]
DUR_TRAMA = 0.02          # 20 ms por trama para calcular energias
DUR_GRABACION = 2.0       # segundos de captura del comando de voz

# Pasos que NO estan en el PDF (apagados = version "solo PDF"):
NORMALIZAR_AMPLITUD = False
DESCARTAR_SILENCIO = False    # ignorar tramas con poca energia
UMBRAL_SILENCIO = 0.05        # fraccion de la energia maxima (si DESCARTAR_SILENCIO)
# ---------------------------------------------------------


def omega_c(fc, fs):
    """Omega_c = 2*pi*fc/fs"""
    return 2 * np.pi * fc / fs


def h_pasabanda_ideal(fc1, fc2, fs, N):
    """
    hBP(n) = [sen(Wc2 n) - sen(Wc1 n)] / (pi n),   n != 0
           = (Wc2 - Wc1) / pi,                      n = 0
    con -(N-1)/2 <= n <= (N-1)/2
    """
    w1, w2 = omega_c(fc1, fs), omega_c(fc2, fs)
    M = (N - 1) // 2
    n = np.arange(-M, M + 1)
    h = np.zeros(N)
    nz = n != 0
    h[nz] = (np.sin(w2 * n[nz]) - np.sin(w1 * n[nz])) / (np.pi * n[nz])
    h[~nz] = (w2 - w1) / np.pi
    return h


def ventana(N, tipo="hamming"):
    """
    Hamming: w(n) = 0.54 - 0.46 cos(2 pi n/(N-1))
    Hanning: w(n) = 0.5  - 0.5  cos(2 pi n/(N-1))     n = 0..N-1
    """
    n = np.arange(N)
    if tipo == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    if tipo == "hanning":
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    raise ValueError("Ventana no soportada")


def disenar_banco(fs=FS, N=N_ORDEN, bandas=BANDAS, tipo=VENTANA):
    """hd(n) = h(n) w(n) para cada subbanda."""
    assert 2 <= len(bandas) <= 5, "El enunciado pide entre 2 y 5 subbandas"
    assert N % 2 == 1, "N debe ser impar"
    return [h_pasabanda_ideal(f1, f2, fs, N) * ventana(N, tipo) for f1, f2 in bandas]


def filtrar(x, hd):
    """y(n) = sum_k h(k) x(n-k)  (convolucion; 'same' compensa el retardo)."""
    return np.convolve(x, hd, mode="same")


def energias_por_trama(y, fs=FS, dur=DUR_TRAMA):
    """Energia de cada trama: E = sum y(n)^2."""
    L = int(dur * fs)
    T = len(y) // L
    tramas = y[: T * L].reshape(T, L)
    return np.sum(tramas ** 2, axis=1)


def cargar_audio(ruta, fs_destino=FS):
    """Lee un audio (valores entre -1 y 1), lo pasa a mono y a fs_destino."""
    x, fs = sf.read(ruta, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != fs_destino:
        g = gcd(fs_destino, fs)
        x = resample_poly(x, fs_destino // g, fs // g)
    return x


def caracteristicas(x, banco, fs=FS, normalizar=None, descartar=None):
    """
    Vector de caracteristicas = [E_prom_1..E_prom_K, Desv_1..Desv_K]
    (energia promedio y desviacion de las energias de cada subbanda).
    """
    normalizar = NORMALIZAR_AMPLITUD if normalizar is None else normalizar
    descartar = DESCARTAR_SILENCIO if descartar is None else descartar

    x = np.asarray(x, dtype=float)
    if normalizar:
        x = x - np.mean(x)
        x = x / (np.max(np.abs(x)) + 1e-12)

    E = np.array([energias_por_trama(filtrar(x, hd), fs) for hd in banco])  # (K, T)

    if descartar:
        total = E.sum(axis=0)
        E = E[:, total > UMBRAL_SILENCIO * total.max()]

    return np.concatenate([E.mean(axis=1), E.std(axis=1)])