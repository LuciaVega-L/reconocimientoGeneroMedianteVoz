import glob, os
from math import gcd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

try:
    from filtrosBanco import FS, disenar_banco, filtrar, energias_por_trama
except ModuleNotFoundError:
    from filtrosBanco import FS, disenar_banco, filtrar, energias_por_trama

CARPETA, CLASES = "entrada", ["hombre", "mujer"]

BANDAS = {
    "4b actual":        [(80, 250), (250, 800), (800, 2000), (2000, 3800)],
    "4b voz-grave":     [(85, 180), (180, 255), (255, 1000), (1000, 3800)],
    "4b formantes":     [(80, 255), (255, 600), (600, 1400), (1400, 3000)],
    "4b uniformes":     [(80, 1000), (1000, 2000), (2000, 3000), (3000, 3900)],
    "3b":               [(80, 300), (300, 1000), (1000, 3800)],
    "5b":               [(80, 160), (160, 300), (300, 800), (800, 2000), (2000, 3800)],
    "2b":               [(80, 1000), (1000, 3800)],
}


def leer_audio(ruta):
    fs, x = wavfile.read(ruta)
    x = x.astype(float)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != FS:
        g = gcd(FS, fs)
        x = resample_poly(x, FS // g, fs // g)
    return x


VENTANA_S = 0.8         


def recortar(x, modo):

    if modo == "completo":
        return x
    W = int(VENTANA_S * FS)
    if len(x) <= W:
        return x
    L = int(0.02 * FS)
    e = energias_por_trama(x, FS)
    centro = int(np.argmax(e)) * L + L // 2
    ini = min(max(0, centro - W // 2), len(x) - W)
    return x[ini:ini + W]


def energias_crudas(x, banco):
    x = x - x.mean()
    x = x / (np.max(np.abs(x)) + 1e-12)
    return np.array([energias_por_trama(filtrar(x, hd), FS) for hd in banco])


def feats(E, umbral):
    # descarta tramas con energia total < umbral * maximo; luego energia promedio y
    # desviacion de las energias de cada subbanda
    total = E.sum(axis=0)
    E = E[:, total > umbral * total.max()]
    return np.concatenate([E.mean(axis=1), E.std(axis=1)])


def error_loo(X, y):
    mal = np.zeros(2, int)
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        cent = np.array([((X[tr] - mu) / sd)[y[tr] == c].mean(0) for c in (0, 1)])
        p = np.argmin(np.linalg.norm(cent - (X[i] - mu) / sd, axis=1))
        if p != y[i]:
            mal[y[i]] += 1
    return mal

audios, y = [], []
for c, clase in enumerate(CLASES):
    for ruta in sorted(glob.glob(os.path.join(CARPETA, clase, "*.wav"))):
        audios.append(leer_audio(ruta)); y.append(c)
y = np.array(y)
print(f"{len(y)} audios | hombre={np.sum(y==0)} mujer={np.sum(y==1)}\n")

res = []
for modo in ("completo", "ventana"):
    recs = [recortar(x, modo) for x in audios]
    for nombre, bandas in BANDAS.items():
        banco = disenar_banco(FS, 255, bandas, "hamming")
        Es = [energias_crudas(x, banco) for x in recs]
        for umbral in (0.05, 0.10, 0.20, 0.30):
            X = np.array([feats(E, umbral) for E in Es])
            mal = error_loo(X, y)
            res.append((100 * mal.sum() / len(y), nombre, modo, umbral, mal))

res.sort(key=lambda r: r[0])
print(f"{'error%':>7} | {'bandas':<14} | {'recorte':<8} | {'umbral':>6} | fallan (hombre, mujer)")
for e, nom, modo, um, mal in res[:12]:
    print(f"{e:7.1f} | {nom:<14} | {modo:<8} | {um:>6.2f} | ({mal[0]}, {mal[1]})")