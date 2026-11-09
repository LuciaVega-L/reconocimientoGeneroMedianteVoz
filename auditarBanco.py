# auditar_banco.py
import glob, os, hashlib
import numpy as np
from filtrosBanco import FS, disenar_banco, caracteristicas, cargar_audio

CARPETA = "entrada"
CLASES = ["hombre", "mujer"]
banco = disenar_banco()

X, y, rutas, personas = [], [], [], []

for c, clase in enumerate(CLASES):
    for ruta in sorted(glob.glob(os.path.join(CARPETA, clase, "*.wav"))):
        h = hashlib.md5(open(ruta, "rb").read()).hexdigest()
        try:
            X.append(caracteristicas(cargar_audio(ruta), banco, FS))
            y.append(c)
            rutas.append(ruta)
            personas.append(clase + "_" + h)
        except Exception as e:
            print(f"Error procesando {ruta}: {e}")

X, y, personas = np.array(X), np.array(y), np.array(personas)


def ajustar(X_tr, y_tr):
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-12
    Z = (X_tr - mu) / sd
    cent = np.array([Z[y_tr == c].mean(0) for c in range(len(CLASES))])
    return mu, sd, cent


def predecir(x, mu, sd, cent):
    z = (x - mu) / sd
    return int(np.argmin(np.linalg.norm(cent - z, axis=1)))


print("\n--- AUDIOS MAL CLASIFICADOS ---")
fallos = 0
for i, p in enumerate(personas):
    te = personas == p
    mu, sd, cent = ajustar(X[~te], y[~te])
    pred = predecir(X[i], mu, sd, cent)
    if pred != y[i]:
        fallos += 1
        print(f"[{fallos}] Esperado: {CLASES[y[i]]:<6} | Predicho: {CLASES[pred]:<6} | Archivo: {rutas[i]}")

print(f"\nTotal fallos: {fallos} de {len(y)} ({100*fallos/len(y):.1f} % error)")