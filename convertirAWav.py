"""
Convierte a .wav (mono, 8000 Hz) todos los audios que NO sean .wav:
.opus, .mp3, .m4a, .ogg, .aac, .flac, .amr, .wma, y archivos SIN extension.
Los .wav existentes no se tocan. Los originales se conservan.

Uso:
    python convertir_a_wav.py                      (usa entrada/hombre y entrada/mujer)
    python convertir_a_wav.py carpeta1 carpeta2    (carpetas que tu elijas)
"""
import os, sys, subprocess
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()       # ffmpeg incluido, no hay que instalarlo
FS = 8000
IGNORAR = {".txt", ".ini", ".db", ".py", ".md", ".csv", ".json", ".ds_store", ".tsv"}

carpetas = sys.argv[1:] or [os.path.join("entrada", "hombre"), os.path.join("entrada", "mujer")]

ok = fallos = 0
for carpeta in carpetas:
    if not os.path.isdir(carpeta):
        print(f"[!] No existe la carpeta: {carpeta}")
        continue
    for nombre in sorted(os.listdir(carpeta)):
        ruta = os.path.join(carpeta, nombre)
        base, ext = os.path.splitext(nombre)
        if not os.path.isfile(ruta) or ext.lower() == ".wav" or ext.lower() in IGNORAR:
            continue
        if nombre.startswith("."):
            continue
        salida = os.path.join(carpeta, base + ".wav")
        if os.path.exists(salida):
            print(f"[=] Ya existe {salida}, se omite")
            continue
        r = subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-i", ruta,
             "-vn", "-ac", "1", "-ar", str(FS), "-c:a", "pcm_s16le", salida],
            capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(salida):
            print(f"[OK] {ruta}  ->  {salida}")
            ok += 1
        else:
            print(f"[X] No pude convertir {ruta}")
            if os.path.exists(salida):
                os.remove(salida)
            fallos += 1

print(f"\nConvertidos: {ok} | Fallidos: {fallos}")