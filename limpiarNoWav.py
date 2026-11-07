
import os, sys

carpetas = sys.argv[1:] or [os.path.join("entrada", "hombre"), os.path.join("entrada", "mujer")]

a_borrar, sin_wav = [], []
for carpeta in carpetas:
    if not os.path.isdir(carpeta):
        print(f"[!] No existe la carpeta: {carpeta}")
        continue
    for nombre in sorted(os.listdir(carpeta)):
        ruta = os.path.join(carpeta, nombre)
        base, ext = os.path.splitext(nombre)
        if not os.path.isfile(ruta) or ext.lower() == ".wav" or nombre.startswith("."):
            continue
        if os.path.exists(os.path.join(carpeta, base + ".wav")):
            a_borrar.append(ruta)
        else:
            sin_wav.append(ruta)

if sin_wav:
    print("NO se borraran (no tienen version .wav, ejecuta convertir_a_wav.py):")
    for r in sin_wav:
        print("  ", r)

if not a_borrar:
    print("No hay nada para borrar.")
    sys.exit()

print(f"\nSe borraran {len(a_borrar)} archivos:")
for r in a_borrar:
    print("  ", r)

if input("\nEscribe SI para confirmar: ").strip().upper() == "SI":
    for r in a_borrar:
        os.remove(r)
    print("Listo. Solo quedan los .wav.")
else:
    print("Cancelado, no se borro nada.")