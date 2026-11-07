
import glob, os
import numpy as np

try:
    from filtrosBanco import FS, disenar_banco, caracteristicas, cargar_audio
except ModuleNotFoundError:
    from filtrosBanco import FS, disenar_banco, caracteristicas, cargar_audio

CARPETA = "entrada"
CLASES = ["hombre", "mujer"]
banco = disenar_banco()


X, y, personas = [], [], []
for c, clase in enumerate(CLASES):
    rutas = sorted(glob.glob(os.path.join(CARPETA, clase, "*.wav")))
    if not rutas:
        print(f"[!] No hay .wav en {os.path.join(CARPETA, clase)}")
    for ruta in rutas:
        try:
            X.append(caracteristicas(cargar_audio(ruta), banco, FS))
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
Z = (X - mu) / sd
dist_ref = float(np.median([np.linalg.norm(Z[i] - cent[y[i]]) for i in range(len(y))]))
np.savez("modelo.npz", mu=mu, sd=sd, cent=cent, dist_ref=dist_ref)
print(f"Distancia tipica de un audio del banco a su centroide: {dist_ref:.2f}")
print("Modelo guardado en modelo.npz")