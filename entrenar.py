
import glob, os
import numpy as np
from scipy.io import wavfile
from filtrosBanco import FS, disenar_banco, caracteristicas

CLASES = ["hombre", "mujer"]
banco = disenar_banco()

X, y, personas = [], [], []
for c, clase in enumerate(CLASES):
    for ruta in sorted(glob.glob(f"banco/{clase}/*.wav")):
        fs, x = wavfile.read(ruta)
        assert fs == FS, f"{ruta}: fs={fs}, se esperaba {FS}"
        X.append(caracteristicas(x.astype(float), banco, fs))
        y.append(c)
        personas.append(clase + "_" + os.path.basename(ruta).rsplit("_", 1)[0])
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