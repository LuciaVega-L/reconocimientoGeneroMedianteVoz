import numpy as np
import sounddevice as sd
from filtrosBanco import FS, DUR_GRABACION, disenar_banco, caracteristicas

CLASES = ["hombre", "mujer"]
m = np.load("modelo.npz")
banco = disenar_banco()

while True:
    input("ENTER y diga la palabra...")
    x = sd.rec(int(DUR_GRABACION * FS), samplerate=FS, channels=1, dtype="float32")
    sd.wait()
    f = caracteristicas(x[:, 0], banco, FS)
    z = (f - m["mu"]) / m["sd"]
    d = np.linalg.norm(m["cent"] - z, axis=1)
    print(f"Genero: {CLASES[int(np.argmin(d))].upper()}   (dist: {d.round(2)})\n")