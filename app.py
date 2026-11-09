import threading
from math import gcd

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from filtrosBanco import (FS, DUR_GRABACION, BANDAS, disenar_banco, caracteristicas,
                          cargar_audio)

CLASES = ["hombre", "mujer"]
DESCARTE = 0.3

PAGE, CARD, BORDE = "#f5f7fb", "#ffffff", "#e5e7eb"
FG, FG_SUAVE = "#1f2933", "#6b7280"
ACENTO, ACENTO_OSC = "#2563eb", "#1d4ed8"
COL_H, COL_M = "#2563eb", "#db2777"
TINTE_H, TINTE_M, TINTE_N = "#dbeafe", "#fce7f3", "#eef2f7"
FUENTE = "Segoe UI"

try:
    modelo = np.load("modelo.npz")
except Exception as e:
    print(f"Error cargando modelo.npz: {e}")
    modelo = {}

BANCO = disenar_banco()


def clasificar(x):
    f = caracteristicas(x, BANCO, FS)
    z = (f - modelo["mu"]) / modelo["sd"]
    d = np.linalg.norm(modelo["cent"] - z, axis=1)
    return int(np.argmin(d)), d, f


def _estilo_ejes(fig, axes):
    fig.patch.set_facecolor(CARD)
    for ax in axes:
        ax.set_facecolor(CARD)
        ax.tick_params(colors=FG_SUAVE, labelsize=9)
        ax.xaxis.label.set_color(FG_SUAVE)
        ax.yaxis.label.set_color(FG_SUAVE)
        ax.grid(True, color="#eef0f3", linewidth=0.8)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color("#d1d5db")


def dibujar(fig, x, clase, f_cap):
    fig.clear()
    k_num = len(BANDAS)

    # ---------------- GRÁFICA 1: Señal capturada x(t) ----------------
    ax1 = fig.add_subplot(2, 1, 1)
    t = np.arange(len(x)) / FS
    col_pred = COL_H if clase == 0 else COL_M
    ax1.plot(t, x, color=col_pred, linewidth=0.8)
    ax1.set_title("1. Señal capturada x(t) (Dominio del Tiempo)", color=FG, fontsize=11, loc="left", fontweight="bold")
    ax1.set_xlabel("Tiempo (s)", fontsize=9)
    ax1.set_ylabel("Amplitud", fontsize=9)

    # ---------------- GRÁFICA 2: Espectro z = |FFT(x)| (Fiel al Tablero) ----------------
    ax2 = fig.add_subplot(2, 1, 2)
    
    # Pseudocódigo del tablero: y = fft(x); z = abs(y)
    N_fft = len(x)
    fft_val = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(N_fft, d=1/FS)

    # 1. Espectro z de la voz capturada
    ax2.plot(freqs, fft_val, color="#10b981", alpha=0.55, label="z = |FFT(x)| (Capturada)", linewidth=1.0)

    # 2. Curvas de referencia Tipo A (Hombre) y Tipo B (Mujer) evaluadas en los centros de subbanda
    e_h = modelo["e_mean_h"] if "e_mean_h" in modelo.files else f_cap[:k_num]
    e_m = modelo["e_mean_m"] if "e_mean_m" in modelo.files else f_cap[:k_num]
    
    centros_banda = [(a + b) / 2 for a, b in BANDAS]
    # Escalamiento visual para superponer las curvas sobre el espectro FFT
    escala = fft_val.max() / (max(e_h.max(), e_m.max()) + 1e-12)
    
    ax2.plot(centros_banda, e_h * escala, 'o--', color=COL_H, label="Tipo A (Hombre)", linewidth=2.0)
    ax2.plot(centros_banda, e_m * escala, 's--', color=COL_M, label="Tipo B (Mujer)", linewidth=2.0)

    # 3. Líneas verticales punteadas de corte k (Límites de subbandas del tablero)
    for a, b in BANDAS:
        ax2.axvline(x=a, color="#6b7280", linestyle="--", linewidth=1.2)
    ax2.axvline(x=BANDAS[-1][1], color="#6b7280", linestyle="--", linewidth=1.2)

    ax2.set_title("2. Espectro z = |FFT(x)| con curvas Tipo A, Tipo B y cortes k", color=FG, fontsize=11, loc="left", fontweight="bold")
    ax2.set_xlabel("Frecuencia (Hz) / Muestras k", fontsize=9)
    ax2.set_ylabel("Espectro z", fontsize=9)
    ax2.legend(loc="upper right", fontsize=8)

    _estilo_ejes(fig, (ax1, ax2))
    fig.tight_layout(pad=1.8, h_pad=2.2)


class VentanaGraficas(tk.Toplevel):
    def __init__(self, master, alcerrar):
        super().__init__(master)
        self.title("Análisis Espectral y Temporal")
        self.configure(bg=CARD)
        self.geometry("1100x750")
        self.minsize(750, 500)
        self.protocol("WM_DELETE_WINDOW", self._cerrar)
        self._alcerrar = alcerrar

        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        try:
            barra = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
            barra.update()
            barra.pack(side="bottom", fill="x")
        except TypeError:
            NavigationToolbar2Tk(self.canvas, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    def actualizar(self, x, clase, f_cap):
        self.deiconify()
        self.lift()
        self.update_idletasks()
        dibujar(self.fig, x, clase, f_cap)
        self.canvas.draw_idle()

    def _cerrar(self):
        self._alcerrar()
        self.destroy()


class App:
    def __init__(self, root):
        self.root = root
        root.title("Reconocimiento de Género por Voz")
        root.configure(bg=PAGE)
        root.geometry("900x480")
        root.minsize(760, 450)
        self.ventana = None
        self.ultimo = None
        self._estilos()

        enc = tk.Frame(root, bg=PAGE)
        enc.pack(fill="x", padx=24, pady=(20, 4))
        tk.Label(enc, text="Reconocimiento de género por voz", bg=PAGE, fg=FG,
                 font=(FUENTE, 18, "bold")).pack(anchor="w")
        tk.Label(enc, text=f"Análisis de banco de filtros · {len(BANDAS)} subbandas · fs = {FS} Hz",
                 bg=PAGE, fg=FG_SUAVE, font=(FUENTE, 10)).pack(anchor="w")

        barra = tk.Frame(root, bg=PAGE)
        barra.pack(fill="x", padx=24, pady=(14, 10))
        self.btn_grabar = ttk.Button(barra, text=f"● Grabar ({DUR_GRABACION:.0f} s)",
                                     style="Primario.TButton", command=self.grabar)
        self.btn_grabar.pack(side="left")
        ttk.Button(barra, text="Cargar .wav", style="Secundario.TButton",
                   command=self.cargar).pack(side="left", padx=10)
        self.btn_graf = ttk.Button(barra, text="Ver gráficas", style="Secundario.TButton",
                                   command=self.abrir_graficas, state="disabled")
        self.btn_graf.pack(side="left")

        res = self._tarjeta(root)
        res.pack(fill="x", padx=24, pady=(0, 20))
        interior = tk.Frame(res, bg=CARD)
        interior.pack(fill="x", padx=20, pady=18)

        tk.Label(interior, text="RESULTADO DEL RECONOCIMIENTO", bg=CARD, fg=FG_SUAVE,
                 font=(FUENTE, 9, "bold")).pack(anchor="w")
        self.badge = tk.Label(interior, text="Sin resultado", bg=TINTE_N, fg=FG_SUAVE,
                                font=(FUENTE, 26, "bold"), padx=26, pady=8)
        self.badge.pack(anchor="w", pady=(6, 4))
        self.lbl_info = tk.Label(interior, text="Presione «Grabar» y pronuncie la palabra seleccionada.",
                                 bg=CARD, fg=FG_SUAVE, font=(FUENTE, 10))
        self.lbl_info.pack(anchor="w")

        chips = tk.Frame(interior, bg=CARD)
        chips.pack(fill="x", pady=(12, 0))
        self.v_h = self._chip(chips, "Distancia a Hombre")
        self.v_m = self._chip(chips, "Distancia a Mujer")
        self.v_dif = self._chip(chips, "Diferencia")
        self.v_pico = self._chip(chips, "Nivel pico")

        if "error_pct" in modelo.files:
            ep = float(modelo["error_pct"])
            txt = (f"Margen de error estimado del sistema: {ep:.1f} % "
                   f"({int(modelo['errores'])} de {int(modelo['total'])} audios mal clasificados) · Meta Lab 2 ≤ 5 %")
            col = "#15803d" if ep <= 5.0 else "#b45309"
        else:
            txt, col = "Margen de error no registrado en modelo.npz", FG_SUAVE
        tk.Label(interior, text=txt, bg=CARD, fg=col, font=(FUENTE, 10, "bold")).pack(anchor="w", pady=(12, 0))

    def _estilos(self):
        st = ttk.Style()
        st.theme_use("clam")
        base = dict(font=(FUENTE, 11, "bold"), padding=(18, 9), borderwidth=0,
                    relief="flat", focuscolor=PAGE)
        st.configure("Primario.TButton", background=ACENTO, foreground="#ffffff", **base)
        st.map("Primario.TButton", background=[("disabled", "#93c5fd"), ("active", ACENTO_OSC)])
        st.configure("Secundario.TButton", background=CARD, foreground=FG, bordercolor=BORDE, **{**base, "borderwidth": 1})
        st.map("Secundario.TButton", background=[("active", "#f3f4f6")])

    def _tarjeta(self, parent):
        return tk.Frame(parent, bg=CARD, highlightbackground=BORDE, highlightthickness=1)

    def _chip(self, parent, titulo):
        caja = tk.Frame(parent, bg="#f3f4f6")
        caja.pack(side="left", padx=(0, 10), ipadx=10, ipady=4)
        tk.Label(caja, text=titulo, bg="#f3f4f6", fg=FG_SUAVE, font=(FUENTE, 8)).pack(anchor="w", padx=6)
        valor = tk.Label(caja, text="—", bg="#f3f4f6", fg=FG, font=(FUENTE, 12, "bold"))
        valor.pack(anchor="w", padx=6)
        return valor

    def grabar(self):
        self.btn_grabar.state(["disabled"])
        self.badge.config(text="Escuchando…", bg="#e0e7ff", fg=ACENTO)
        self.lbl_info.config(text=f"Hable ahora ({DUR_GRABACION:.0f} s)...")
        threading.Thread(target=self._grabar_hilo, daemon=True).start()

    def _grabar_hilo(self):
        try:
            import sounddevice as sd
            from scipy.signal import resample_poly
            try:
                sd.check_input_settings(samplerate=FS, channels=1)
                fs_mic = FS
            except Exception:
                fs_mic = 44100
            x = sd.rec(int((DUR_GRABACION + DESCARTE) * fs_mic), samplerate=fs_mic, channels=1, dtype="float32")
            sd.wait()
            x = x[int(DESCARTE * fs_mic):, 0]
            if fs_mic != FS:
                g = gcd(FS, fs_mic)
                x = resample_poly(x, FS // g, fs_mic // g)
            self.root.after(0, self.mostrar, x)
        except Exception as e:
            self.root.after(0, self.error, str(e))

    def cargar(self):
        ruta = filedialog.askopenfilename(filetypes=[("Audio WAV", "*.wav")])
        if ruta:
            self.mostrar(cargar_audio(ruta))

    def mostrar(self, x):
        self.btn_grabar.state(["!disabled"])
        clase, d, f_cap = clasificar(x)
        color, tinte = (COL_H, TINTE_H) if clase == 0 else (COL_M, TINTE_M)
        self.badge.config(text=CLASES[clase].upper(), bg=tinte, fg=color)
        self.lbl_info.config(text="Género clasificado correctamente.")
        self.v_h.config(text=f"{d[0]:.2f}")
        self.v_m.config(text=f"{d[1]:.2f}")
        self.v_dif.config(text=f"{abs(d[0] - d[1]):.2f}")
        self.v_pico.config(text=f"{20 * np.log10(np.max(np.abs(x)) + 1e-12):.1f} dBFS")
        self.ultimo = (x, clase, f_cap)
        self.btn_graf.state(["!disabled"])
        self.abrir_graficas()

    def abrir_graficas(self):
        if self.ultimo is None:
            return
        if self.ventana is None:
            self.ventana = VentanaGraficas(self.root, self._ventana_cerrada)
        self.ventana.actualizar(*self.ultimo)

    def _ventana_cerrada(self):
        self.ventana = None

    def error(self, msg):
        self.btn_grabar.state(["!disabled"])
        self.badge.config(text="Error", bg="#fee2e2", fg="#dc2626")
        self.lbl_info.config(text=msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()