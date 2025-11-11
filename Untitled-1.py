
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- 1. Definición de Parámetros y Señales ---
Fs = 1000       # Frecuencia de muestreo (Hz)
T = 1.0 / Fs    # Período de muestreo
L = 1500        # Número de muestras
t = np.arange(L) * T # Vector de tiempo

# Señal Deseada (Baja Frecuencia)
f_deseada = 5   # 5 Hz
señal_deseada = 0.7 * np.sin(2 * np.pi * f_deseada * t)

# Ruido (Alta Frecuencia)
f_ruido = 100   # 100 Hz
ruido = 0.3 * np.sin(2 * np.pi * f_ruido * t)

# Ruido Blanco
ruido_blanco = 0.1 * np.random.normal(size=L)

# Señal Compuesta de Entrada
señal_entrada = señal_deseada + ruido + ruido_blanco

# --- Gráfico: Señal Antes de Aplicar el Filtro ---
plt.figure(figsize=(10, 4))
plt.plot(t, señal_entrada, label='Señal Ruidosa')
plt.title('Señal de Entrada Compuesta (5 Hz + Ruido)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.xlim(0, 0.1) # Zoom
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- 2. Diseño IIR Butterworth Pasa Bajos ---
orden = 5           # Orden del filtro
f_c = 20.0          # Frecuencia de corte (Hz)
Wn_norm = f_c / (Fs / 2) # Frecuencia de corte normalizada (respecto a Nyquist)

# Coeficientes del Filtro IIR
b_lpf_iir, a_lpf_iir = signal.butter(orden, Wn_norm, btype='low', analog=False) 

# Aplicación del Filtro (Usando filtfilt para fase cero)
señal_filtrada_iir = signal.filtfilt(b_lpf_iir, a_lpf_iir, señal_entrada)

# --- Análisis de la Respuesta en Frecuencia IIR ---
w, h = signal.freqz(b_lpf_iir, a_lpf_iir, fs=Fs)

plt.figure(figsize=(10, 4))
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.title('Respuesta en Frecuencia (IIR Butterworth Pasa Bajos)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Ganancia (dB)')
plt.axvline(f_c, color='r', linestyle='--', label=f'f_c = {f_c} Hz')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Comparación IIR ---
plt.figure(figsize=(10, 4))
plt.plot(t, señal_entrada, 'r', alpha=0.5, label='Señal Ruidosa', linewidth=1)
plt.plot(t, señal_filtrada_iir, 'b', label='Señal Filtrada (LPF IIR)', linewidth=2)
plt.title('Comparación: Señal Original vs. Filtrada (LPF IIR)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.xlim(0, 0.1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- 3. Diseño FIR con Ventana (Hamming) Pasa Bajos ---
orden_fir = 50      # Orden (taps) del filtro FIR. Mayor orden = mejor selectividad
f_c_fir = 20.0      # Mantenemos la misma frecuencia de corte

# Coeficientes del Filtro FIR (usando Ventana de Hamming)
b_lpf_fir = signal.firwin(orden_fir, f_c_fir, fs=Fs, pass_zero='lowpass', window='hamming')
a_lpf_fir = 1.0     # Para FIR, el denominador es 1

# Aplicación del Filtro
# 'filtfilt' también se puede usar en FIR para eliminar el retardo.
señal_filtrada_fir = signal.filtfilt(b_lpf_fir, a_lpf_fir, señal_entrada)

# --- Comparación FIR ---
plt.figure(figsize=(10, 4))
plt.plot(t, señal_filtrada_iir, 'b', label='Señal Filtrada (LPF IIR)', linewidth=2, alpha=0.6)
plt.plot(t, señal_filtrada_fir, 'g', label='Señal Filtrada (LPF FIR)', linewidth=1)
plt.title('Comparación de Resultados (IIR vs FIR)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.xlim(0, 0.1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()