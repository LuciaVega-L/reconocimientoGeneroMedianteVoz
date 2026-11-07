"""Captura en tiempo real (microfono) y reconoce el genero.
Requiere haber corrido antes:  python entrenar.py  (genera modelo.npz)
"""
from math import gcd
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

try:
    from filtrosBanco import FS, DUR_GRABACION, disenar_banco, caracteristicas
except ModuleNotFoundError:
    from filtrosBanco import FS, DUR_GRABACION, disenar_banco, caracteristicas

FS_MIC = 44100            # la mayoria de microfonos de PC trabajan bien a 44100 Hz
CLASES = ["hombre", "mujer"]

m = np.load("modelo.npz")
banco = disenar_banco()
g = gcd(FS, FS_MIC)

print("Dispositivo de entrada:", sd.query_devices(kind="input")["name"])
while True:
    entrada = input("\nENTER para grabar (q para salir)... ")
    if entrada.strip().lower() == "q":
        break
    print(f"Hable ahora ({DUR_GRABACION:.0f} s)...")
    x = sd.rec(int(DUR_GRABACION * FS_MIC), samplerate=FS_MIC, channels=1, dtype="float32")
    sd.wait()
    x = resample_poly(x[:, 0], FS // g, FS_MIC // g)        # 44100 Hz -> 8000 Hz
    f = caracteristicas(x, banco, FS)
    z = (f - m["mu"]) / m["sd"]
    d = np.linalg.norm(m["cent"] - z, axis=1)
    print(f"Genero: {CLASES[int(np.argmin(d))].upper()}   "
          f"(distancia a hombre={d[0]:.2f}, a mujer={d[1]:.2f})")