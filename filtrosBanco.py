from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# ---------------- PARAMETROS (Laboratorio 2) ----------------
FS = 8000                 # Frecuencia de muestreo (Hz)[cite: 65, 78]
N_ORDEN = 255             # Orden del FIR (impar, N-1=254)[cite: 62, 65]
VENTANA = "hamming"       # Ventana Hamming (PDF pág. 17, 29)[cite: 63, 75]
BANDAS = [                # 5 subbandas iguales de 700 Hz (PDF pág. 44, 78)
    (100, 800),
    (800, 1500),
    (1500, 2200),
    (2200, 2900),
    (2900, 3600),
]
DUR_TRAMA = 0.02          # Tramas de 20 ms[cite: 44]
DUR_GRABACION = 2.0       # Duración en segundos[cite: 44, 51]

NORMALIZAR_AMPLITUD = True
DESCARTAR_SILENCIO = True
UMBRAL_SILENCIO = 0.05
# -----------------------------------------------------------


def omega_c(fc, fs):
    """Frecuencia digital Omega_c = 2*pi*fc/fs (PDF pág. 16, 29)."""
    return 2 * np.pi * fc / fs


def h_pasabanda_ideal(fc1, fc2, fs, N):
    """
    Filtro pasabanda ideal (PDF pág. 19, 29):
    hBP(n) = [sen(Wc2 n) - sen(Wc1 n)] / (pi n),  n != 0
           = (Wc2 - Wc1) / pi,                  n = 0
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
    """Ventana Hamming: w(n) = 0.54 - 0.46 cos(2 pi n/(N-1)) (PDF pág. 17, 29)."""
    n = np.arange(N)
    if tipo == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    if tipo == "hanning":
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    raise ValueError("Ventana no soportada")


def verificar_bandas(bandas=BANDAS, fs=FS, N=N_ORDEN, tol=1e-9):
    anchos = [f2 - f1 for f1, f2 in bandas]
    assert max(anchos) - min(anchos) <= tol, f"Anchos distintos: {anchos}"
    assert all(bandas[i][1] <= bandas[i + 1][0] for i in range(len(bandas) - 1)), "Subbandas solapadas"
    assert bandas[-1][1] < fs / 2, "fc2 debe ser menor que Nyquist (fs/2)"[cite: 1]
    trans = 3.3 * fs / N
    assert min(anchos) > trans, f"BW={min(anchos)} Hz <= transicion de la ventana ({trans:.0f} Hz)"
    return anchos


def disenar_banco(fs=FS, N=N_ORDEN, bandas=BANDAS, tipo=VENTANA):
    """hd(n) = hBP(n) * w(n) para cada subbanda (PDF pág. 16, 29)."""
    assert 2 <= len(bandas) <= 5, "El laboratorio exige entre 2 y 5 subbandas"[cite: 45, 78]
    assert N % 2 == 1, "N debe ser impar"[cite: 62]
    verificar_bandas(bandas, fs, N)
    return [h_pasabanda_ideal(f1, f2, fs, N) * ventana(N, tipo) for f1, f2 in bandas]


def filtrar(x, hd):
    """y(n) = sum_k h(k) x(n-k) (Convolución discreta; PDF pág. 7, 30)."""
    return np.convolve(x, hd, mode="same")


def energias_por_trama(y, fs=FS, dur=DUR_TRAMA):
    """Energía por trama: E = sum y(n)^2 (PDF pág. 28, 45)."""
    L = int(dur * fs)
    T = len(y) // L
    if T == 0:
        return np.array([np.sum(y**2)])
    tramas = y[: T * L].reshape(T, L)
    return np.sum(tramas ** 2, axis=1)


def cargar_audio(ruta, fs_destino=FS):
    x, fs = sf.read(ruta, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != fs_destino:
        g = gcd(fs_destino, fs)
        x = resample_poly(x, fs_destino // g, fs // g)
    return x


def caracteristicas(x, banco, fs=FS, normalizar=None, descartar=None):
    """
    Vector de características = [E_prom_1..K, Desv_1..K]
    (Energía promedio y desviación estándar por subbanda; PDF pág. 46 y Lab 2).
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
        mask = total > UMBRAL_SILENCIO * total.max()
        if np.any(mask):
            E = E[:, mask]

    return np.concatenate([E.mean(axis=1), E.std(axis=1)])