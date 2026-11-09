import glob, os, hashlib
import numpy as np

try:
    from filtrosBanco import FS, BANDAS, disenar_banco, caracteristicas, cargar_audio
except ModuleNotFoundError:
    from filtrosBanco import FS, BANDAS, disenar_banco, caracteristicas, cargar_audio

CARPETA = "entrada"
CLASES = ["hombre", "mujer"]
banco = disenar_banco()
k_bandas = len(BANDAS)

X, y, personas = [], [], []
for c, clase in enumerate(CLASES):
    rutas = sorted(glob.glob(os.path.join(CARPETA, clase, "*.wav")))
    for ruta in rutas:
        h = hashlib.md5(open(ruta, "rb").read()).hexdigest()
        try:
            X.append(caracteristicas(cargar_audio(ruta), banco, FS))
            y.append(c)
            personas.append(clase + "_" + h)
        except Exception as e:
            continue

X, y, personas = np.array(X), np.array(y), np.array(personas)


def ajustar(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    cent = np.array([Z[y == c].mean(0) for c in range(len(CLASES))])
    return mu, sd, cent


def predecir(x, mu, sd, cent):
    z = (x - mu) / sd
    return int(np.argmin(np.linalg.norm(cent - z, axis=1)))


# Validación Leave-One-Person-Out
errores = 0
for p in np.unique(personas):
    te = personas == p
    mu, sd, cent = ajustar(X[~te], y[~te])
    errores += sum(predecir(x, mu, sd, cent) != yi for x, yi in zip(X[te], y[te]))

error_pct = 100 * errores / len(y) if len(y) > 0 else 0
print(f"Error estimado: {error_pct:.1f} % ({errores} de {len(y)} audios mal clasificados)")

mu, sd, cent = ajustar(X, y)
Z = (X - mu) / sd
dist_ref = float(np.median([np.linalg.norm(Z[i] - cent[y[i]]) for i in range(len(y))]))

# Promedios de energía original por subbanda (para las gráficas comparativas del tablero)
e_mean_h = X[y == 0][:, :k_bandas].mean(axis=0)
e_mean_m = X[y == 1][:, :k_bandas].mean(axis=0)

np.savez("modelo.npz", mu=mu, sd=sd, cent=cent, dist_ref=dist_ref,
         error_pct=error_pct, errores=errores, total=len(y),
         e_mean_h=e_mean_h, e_mean_m=e_mean_m)

print("Modelo actualizado exitosamente con perfiles de Hombre y Mujer.")