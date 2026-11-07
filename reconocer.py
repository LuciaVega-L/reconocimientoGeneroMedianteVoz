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

try:                      # si el microfono acepta 8000 Hz se graba directo a FS
    sd.check_input_settings(samplerate=FS, channels=1)
    FS_MIC = FS
except Exception:         # si no, se graba a 44100 Hz y se remuestrea
    FS_MIC = 44100
DESCARTE = 0.3            # s iniciales que se descartan (ruido del ENTER / arranque del mic)
CLASES = ["hombre", "mujer"]

m = np.load("modelo.npz")
ref = float(m["dist_ref"]) if "dist_ref" in m.files else None
banco = disenar_banco()
g = gcd(FS, FS_MIC)

print("Dispositivo de entrada:", sd.query_devices(kind="input")["name"], f"| grabando a {FS_MIC} Hz")
while True:
    if input("\nENTER para grabar (q para salir)... ").strip().lower() == "q":
        break
    print(f"Hable ahora ({DUR_GRABACION:.0f} s)...")
    x = sd.rec(int((DUR_GRABACION + DESCARTE) * FS_MIC), samplerate=FS_MIC, channels=1, dtype="float32")
    sd.wait()
    x = x[int(DESCARTE * FS_MIC):, 0]
    if FS_MIC != FS:
        x = resample_poly(x, FS // g, FS_MIC // g)           # 44100 Hz -> 8000 Hz

    pico = np.max(np.abs(x)) + 1e-12
    print(f"Nivel pico: {20*np.log10(pico):.1f} dBFS  (ideal: entre -20 y -3)")

    f = caracteristicas(x, banco, FS)
    z = (f - m["mu"]) / m["sd"]
    d = np.linalg.norm(m["cent"] - z, axis=1)
    print(f"Genero: {CLASES[int(np.argmin(d))].upper()}   "
          f"(distancia a hombre={d[0]:.2f}, a mujer={d[1]:.2f})")
    if ref and d.min() > 1.5 * ref:
        print(f"[!] Grabacion inusual: la distancia ({d.min():.2f}) es mucho mayor que la tipica del banco "
              f"({ref:.2f}). Revisa ruido, volumen, la palabra o el microfono.")