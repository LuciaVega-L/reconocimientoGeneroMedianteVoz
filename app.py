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

modelo = np.load("modelo.npz")
BANCO = disenar_banco()


def clasificar(x):
    f = caracteristicas(x, BANCO, FS)
    z = (f - modelo["mu"]) / modelo["sd"]
    d = np.linalg.norm(modelo["cent"] - z, axis=1)
    k = len(BANCO)
    return int(np.argmin(d)), d, f[:k]


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


def dibujar(fig, x, clase, e_prom):
    fig.clear()
    color = COL_H if clase == 0 else COL_M

    ax1 = fig.add_subplot(2, 1, 1)
    t = np.arange(len(x)) / FS
    ax1.plot(t, x, color=color, linewidth=0.9)
    ax1.fill_between(t, x, 0, color=color, alpha=0.12)
    ax1.set_xlim(0, t[-1])
    ax1.set_title("Señal de voz capturada", color=FG, fontsize=14, loc="left", fontweight="bold")
    ax1.set_xlabel("Tiempo (s)", fontsize=11)
    ax1.set_ylabel("Amplitud", fontsize=11)
    ax1.minorticks_on()

    ax2 = fig.add_subplot(2, 1, 2)
    etiquetas = [f"{a}–{b} Hz" for a, b in BANDAS]
    barras = ax2.bar(etiquetas, e_prom, color=color, width=0.45)
    ax2.margins(y=0.15)
    for b, v in zip(barras, e_prom):
        ax2.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", color=FG, fontsize=11)
    ax2.set_title("Energía promedio por subbanda", color=FG, fontsize=14, loc="left", fontweight="bold")
    ax2.set_xlabel("Subbanda", fontsize=11)
    ax2.set_ylabel("Energía promedio", fontsize=11)

    _estilo_ejes(fig, (ax1, ax2))
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=10)
    fig.tight_layout(pad=2.0, h_pad=3.0)


class VentanaGraficas(tk.Toplevel):
    def __init__(self, master, alcerrar):
        super().__init__(master)
        self.title("Gráficas de la señal")
        self.configure(bg=CARD)
        self.geometry("1200x820")
        self.minsize(800, 560)
        self.protocol("WM_DELETE_WINDOW", self._cerrar)
        self._alcerrar = alcerrar

        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        try:
            barra = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
            barra.update()
            barra.pack(side="bottom", fill="x")
        except TypeError:
            NavigationToolbar2Tk(self.canvas, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    def actualizar(self, x, clase, e_prom):
        self.deiconify()
        self.lift()
        self.update_idletasks()
        try:
            dibujar(self.fig, x, clase, e_prom)
        except Exception:
            import traceback
            detalle = traceback.format_exc()
            print(detalle)
            self.fig.clear()
            self.fig.text(0.03, 0.97, "No se pudieron dibujar las gráficas:\n\n" + detalle,
                          va="top", ha="left", fontsize=9, color="#b91c1c", family="monospace")
        self.canvas.draw_idle()

    def _cerrar(self):
        self._alcerrar()
        self.destroy()


class App:
    def __init__(self, root):
        self.root = root
        root.title("Reconocimiento de género por voz")
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
        self.btn_grabar = ttk.Button(barra, text=f"●  Grabar ({DUR_GRABACION:.0f} s)",
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

        tk.Label(interior, text="RESULTADO", bg=CARD, fg=FG_SUAVE,
                 font=(FUENTE, 9, "bold")).pack(anchor="w")
        self.badge = tk.Label(interior, text="Sin resultado", bg=TINTE_N, fg=FG_SUAVE,
                                font=(FUENTE, 26, "bold"), padx=26, pady=8)
        self.badge.pack(anchor="w", pady=(6, 4))
        self.lbl_info = tk.Label(interior, text="Pulsa «Grabar» y di la palabra seleccionada.",
                                 bg=CARD, fg=FG_SUAVE, font=(FUENTE, 10))
        self.lbl_info.pack(anchor="w")

        chips = tk.Frame(interior, bg=CARD)
        chips.pack(fill="x", pady=(12, 0))
        self.v_h = self._chip(chips, "Distancia a hombre")
        self.v_m = self._chip(chips, "Distancia a mujer")
        self.v_dif = self._chip(chips, "Diferencia")
        self.v_pico = self._chip(chips, "Nivel pico")

        if "error_pct" in modelo.files:
            ep = float(modelo["error_pct"])
            txt = (f"Margen de error estimado del sistema: {ep:.1f} %  "
                   f"({int(modelo['errores'])} de {int(modelo['total'])} audios mal clasificados "
                   f"en la validación) · meta ≤ 5 %")
            col = "#15803d" if ep <= 5 else "#b45309"
        else:
            txt, col = "Margen de error: vuelve a correr entrenar.py para calcularlo.", FG_SUAVE
        tk.Label(interior, text=txt, bg=CARD, fg=col, font=(FUENTE, 10, "bold")).pack(anchor="w", pady=(12, 0))

    def _estilos(self):
        st = ttk.Style()
        st.theme_use("clam")
        base = dict(font=(FUENTE, 11, "bold"), padding=(18, 9), borderwidth=0,
                    relief="flat", focuscolor=PAGE)
        st.configure("Primario.TButton", background=ACENTO, foreground="#ffffff", **base)
        st.map("Primario.TButton",
               background=[("disabled", "#93c5fd"), ("active", ACENTO_OSC)],
               foreground=[("disabled", "#ffffff")])
        st.configure("Secundario.TButton", background=CARD, foreground=FG,
                     bordercolor=BORDE, **{**base, "borderwidth": 1})
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
        self.lbl_info.config(text=f"Hable ahora ({DUR_GRABACION:.0f} s)")
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
            x = sd.rec(int((DUR_GRABACION + DESCARTE) * fs_mic), samplerate=fs_mic,
                       channels=1, dtype="float32")
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
        clase, d, e_prom = clasificar(x)
        color, tinte = (COL_H, TINTE_H) if clase == 0 else (COL_M, TINTE_M)
        self.badge.config(text=CLASES[clase].upper(), bg=tinte, fg=color)
        self.lbl_info.config(text="Género detectado a partir de la energía en las subbandas.")
        self.v_h.config(text=f"{d[0]:.2f}")
        self.v_m.config(text=f"{d[1]:.2f}")
        self.v_dif.config(text=f"{abs(d[0] - d[1]):.2f}")
        self.v_pico.config(text=f"{20 * np.log10(np.max(np.abs(x)) + 1e-12):.1f} dBFS")
        self.ultimo = (x, clase, e_prom)
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
    App(root)
    root.mainloop()