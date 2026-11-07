
import argparse, os
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from filtrosBanco import FS, DUR_GRABACION

ap = argparse.ArgumentParser()
ap.add_argument("--genero", choices=["hombre", "mujer"], required=True)
ap.add_argument("--persona", required=True)
ap.add_argument("--tomas", type=int, default=5)
a = ap.parse_args()

carpeta = os.path.join("banco", a.genero)
os.makedirs(carpeta, exist_ok=True)

for i in range(a.tomas):
    input(f"[{i+1}/{a.tomas}] ENTER y diga la palabra elegida...")
    x = sd.rec(int(DUR_GRABACION * FS), samplerate=FS, channels=1, dtype="float32")
    sd.wait()
    ruta = os.path.join(carpeta, f"{a.persona}_{i}.wav")
    wavfile.write(ruta, FS, (x[:, 0] * 32767).astype(np.int16))
    print("guardado:", ruta)