"""Lee los audios de entrada/hombre y entrada/mujer, extrae caracteristicas,
entrena y estima el error.
- El volumen se normaliza dentro de caracteristicas() (quita la media y divide
  por el pico), asi que no hace falta ajustarlo antes.
- Si un audio no esta a 8000 Hz o es estereo, se corrige aqui mismo.
Clasificador provisional: distancia al centroide de cada clase (z-score).
Validacion: dejar una persona fuera (leave-one-person-out).
"""
import glob, os
from math import gcd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

try:
    from filtrosBanco import FS, disenar_banco, caracteristicas
except ModuleNotFoundError:
    from filtrosBanco import FS, disenar_banco, caracteristicas

CARPETA = "entrada"
CLASES = ["hombre", "mujer"]
banco = disenar_banco()


def leer_audio(ruta):
    fs, x = wavfile.read(ruta)
    x = x.astype(float)
    if x.ndim > 1:                       # estereo -> mono
        x = x.mean(axis=1)
    if fs != FS:                         # remuestrear a FS
        g = gcd(FS, fs)
        x = resample_poly(x, FS // g, fs // g)
    return x


X, y, personas = [], [], []
for c, clase in enumerate(CLASES):
    rutas = sorted(glob.glob(os.path.join(CARPETA, clase, "*.wav")))
    if not rutas:
        print(f"[!] No hay .wav en {os.path.join(CARPETA, clase)}")
    for ruta in rutas:
        try:
            X.append(caracteristicas(leer_audio(ruta), banco, FS))
        except Exception as e:
            print(f"[X] Se omite {ruta}: {e}")
            continue
        y.append(c)
        personas.append(clase + "_" + os.path.basename(ruta))   # 1 archivo = 1 persona
X, y, personas = np.array(X), np.array(y), np.array(personas)
print(f"{len(y)} grabaciones | hombre={np.sum(y==0)} mujer={np.sum(y==1)}")


def ajustar(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    cent = np.array([Z[y == c].mean(0) for c in range(len(CLASES))])
    return mu, sd, cent


def predecir(x, mu, sd, cent):
    z = (x - mu) / sd
    return int(np.argmin(np.linalg.norm(cent - z, axis=1)))


# Validacion leave-one-person-out
errores = 0
for p in np.unique(personas):
    te = personas == p
    mu, sd, cent = ajustar(X[~te], y[~te])
    errores += sum(predecir(x, mu, sd, cent) != yi for x, yi in zip(X[te], y[te]))
print(f"Error estimado: {100*errores/len(y):.1f} %  (meta: <= 5 %)")

mu, sd, cent = ajustar(X, y)
np.savez("modelo.npz", mu=mu, sd=sd, cent=cent)
print("Modelo guardado en modelo.npz")