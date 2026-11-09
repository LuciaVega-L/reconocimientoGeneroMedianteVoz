import glob, os, hashlib, shutil
import numpy as np
from filtrosBanco import FS, disenar_banco, caracteristicas, cargar_audio

CARPETA_ENTRADA = "entrada"
CARPETA_CUARENTENA = "cuarentena"
CLASES = ["hombre", "mujer"]

banco = disenar_banco()
X, y, rutas, personas = [], [], [], []

# 1. Cargar y extraer características
for c, clase in enumerate(CLASES):
    for ruta in sorted(glob.glob(os.path.join(CARPETA_ENTRADA, clase, "*.wav"))):
        h = hashlib.md5(open(ruta, "rb").read()).hexdigest()
        try:
            X.append(caracteristicas(cargar_audio(ruta), banco, FS))
            y.append(c)
            rutas.append(ruta)
            personas.append(clase + "_" + h)
        except Exception as e:
            print(f"[X] Error cargando {ruta}: {e}")

X, y, personas = np.array(X), np.array(y), np.array(personas)


def ajustar(X_tr, y_tr):
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-12
    Z = (X_tr - mu) / sd
    cent = np.array([Z[y_tr == c].mean(0) for c in range(len(CLASES))])
    return mu, sd, cent


def predecir(x, mu, sd, cent):
    z = (x - mu) / sd
    return int(np.argmin(np.linalg.norm(cent - z, axis=1)))


# 2. Identificar audios mal clasificados (Leave-One-Person-Out)
audios_fallidos = []
for i, p in enumerate(personas):
    te = personas == p
    mu, sd, cent = ajustar(X[~te], y[~te])
    pred = predecir(X[i], mu, sd, cent)
    if pred != y[i]:
        audios_fallidos.append((rutas[i], CLASES[y[i]], CLASES[pred]))

print(f"\nSe encontraron {len(audios_fallidos)} audios mal clasificados de {len(y)} en total.")

if not audios_fallidos:
    print("El dataset ya cumple con el criterio.")
else:
    # 3. Confirmación antes de mover
    respuesta = input("\n¿Desea mover estos archivos a la carpeta 'cuarentena/' para su revisión? (s/n): ")
    if respuesta.lower() == 's':
        for ruta, real, predicha in audios_fallidos:
            subcarpeta = os.path.join(CARPETA_CUARENTENA, real)
            os.makedirs(subcarpeta, exist_ok=True)
            destino = os.path.join(subcarpeta, os.path.basename(ruta))
            shutil.move(ruta, destino)
            print(f"Movido: {ruta} -> {destino}")
        print("\nArchivos movidos con éxito. Vuelve a ejecutar 'python entrenar.py'.")
    else:
        print("\nOperación cancelada. No se movió ningún archivo.")