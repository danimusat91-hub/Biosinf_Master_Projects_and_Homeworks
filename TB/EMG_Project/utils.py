import numpy as np
from scipy.signal import spectrogram
from scipy.signal import stft
import pywt
from scipy.stats import skew,kurtosis
import scipy.signal
from scipy.signal import welch,periodogram
from scipy.integrate import quad
from scipy import signal
import matplotlib.pyplot as plt




def MAV(data):
    return np.mean(np.abs(data))

# zero crossing rate
def ZCR(data,threshold):
	return ((data[:-1] * data[1:] < 0) & (np.abs(data[:-1] - data[1:]) > threshold)).sum()


#waveform length
def WL(data):
	return  (abs(data[:-1]-data[1:])).sum()


#root mean square
def RMS(data):
    return np.sqrt(np.mean(data**2))
	
#slope sign changes
def SSC(data, alpha):
    # x[i] - x[i-1]
    slope1 = data[1:-1] - data[:-2]
    # x[i] - x[i+1]
    slope2 = data[1:-1] - data[2:]
    
    # A sign change occurs if both slopes have the same sign (product > 0)
    # and the change is significant (> alpha)
    return ((slope1 * slope2) > alpha).sum()

def Energy(data):
    spectral_amplitude = np.abs(np.fft.fft(data))
    spectral_energy = np.sum(spectral_amplitude)
    return spectral_energy

def spectralPower(data):
    spectrum = np.abs(np.fft.fft(data))**2
    total_power = np.sum(spectrum)
    return total_power

def HJ(data):
    fft_result = np.fft.fft(data)
    spectral_amplitude = np.abs(fft_result)
    dominant_frequencies_indices = np.argsort(spectral_amplitude)[::-1][:512]
    dominant_frequencies_values = np.fft.fftfreq(len(data))[dominant_frequencies_indices]
    spectral_energy = np.sum(spectral_amplitude)
    if spectral_energy != 0:
        bandwidth = np.sum(spectral_amplitude[dominant_frequencies_indices]) / spectral_energy
    else:
        bandwidth = 0
    return bandwidth


def Skewness(data):
    return skew(data)



def wavelet(signal):
    fs = 512  # Frecvență de eșantionare (Hz)
    T = 2  # Durata semnalului (secunde)
    t = np.arange(0, T, 1/fs)  # Vector de timp
    # Creați un wavelet mexican hat (Ricker wavelet)
    wavelet = pywt.ContinuousWavelet('mexh')
    # Generați coeficienții wavelet pentru semnal
    coeffs,_ = pywt.cwt(signal, scales=np.arange(1, 25), wavelet=wavelet)
    features = []  
    # Calcularea energiei totale în coeficienții wavelet
    total_energy = np.sum(np.abs(coeffs) ** 2)
    peaks = np.max(coeffs, axis=1)
    return kurtosis(coeffs)




def ampl_filter(data,threshold):
    for i in range(len(data)):
        if data[i] < threshold:
            data[i] = 0
    return data



def iemg(data):
    spectral_amplitude = np.abs(data)
    iemg_data = np.sum(spectral_amplitude)
    return iemg_data

def clean_signal_list(channel_list, fs=512):
    """
    channel_list: List of 8 numpy arrays, each 30720 samples long.
    fs: Sampling frequency (512 Hz).
    """
    cleaned_channels = []
    
    # 1. Band-pass Filter Design (Industry standard 20-250 Hz)
    # This helps in recognizing movements with varying intensity [cite: 91, 92]
    low = 20 / (0.5 * fs)
    high = 150 / (0.5 * fs)
    b_band, a_band = signal.butter(8, [low, high], btype='band')
    
    # 2. Notch Filter Design (to remove 50Hz/60Hz power line noise) [cite: 76, 88]
    # Adjust 60 to 50 depending on your dataset's origin
    w0 = 60 / (0.5 * fs) 
    b_notch, a_notch = signal.iirnotch(w0, Q=60)
    
    for raw_channel in channel_list:
        # A. DC Offset Removal
        # Standard deviation and RMS are equal only when mean is zero
        centered = raw_channel - np.mean(raw_channel)
        
        # # B. Apply Band-pass
        # # Filters noise and artifacts to facilitate feature extraction
        filtered_band = signal.filtfilt(b_band, a_band, centered)
        #
        # # C. Apply Notch
        filtered_signal= signal.filtfilt(b_notch, a_notch, filtered_band)


        cleaned_channels.append(filtered_signal)
        
    return cleaned_channels # Returns list of 8 cleaned arrays


def plot_emg_analysis(signal_data, fs=512, title="sEMG Signal Analysis"):
    """
    signal_data: list or array of shape (8, 30720)
    fs: sampling frequency (512 Hz as calculated)
    """
    num_channels = 8
    time = np.arange(signal_data[0].size) / fs
    
    fig, axes = plt.subplots(num_channels, 2, figsize=(15, 20), constrained_layout=True)
    fig.suptitle(title, fontsize=16)

    for i in range(num_channels):
        ch_data = signal_data[i]
        
        # --- 1. Time Domain Plot ---
        # Plotting first 2 seconds for clarity
        axes[i, 0].plot(time[:fs*2], ch_data[:fs*2], color='tab:blue')
        axes[i, 0].set_title(f"Channel {i+1} - Time Domain")
        axes[i, 0].set_ylabel("Amplitude (mV)")
        axes[i, 0].grid(True, alpha=0.3)
        if i == num_channels - 1:
            axes[i, 0].set_xlabel("Time (s)")

        # --- 2. Frequency Domain Plot (PSD) ---
        # Using Welch method as required [cite: 110]
        freqs, psd = signal.welch(ch_data, fs, nperseg=1024)
        
        axes[i, 1].semilogy(freqs, psd, color='tab:red')
        axes[i, 1].set_title(f"Channel {i+1} - Power Spectral Density")
        axes[i, 1].set_ylabel("PSD (V^2/Hz)")
        axes[i, 1].set_xlim([0, fs/2]) # Plot up to Nyquist [cite: 109]
        axes[i, 1].grid(True, which='both', alpha=0.3)
        if i == num_channels - 1:
            axes[i, 1].set_xlabel("Frequency (Hz)")
    plt.show()

def codeOneHot(Y_int,Kclass):
    DB_size = Y_int.shape[0]
    Y_onehot = np.zeros((DB_size, Kclass))
    for i in range(0, DB_size):
        Y_onehot[i,Y_int[i]] = 1
    return Y_onehot

def getUA(OUT, TAR, Kclass):
    Kclass = OUT.shape[1]
    VN = np.sum(TAR, axis=0)
    aux = TAR - OUT
    WN = np.sum((aux + np.absolute(aux))//2, axis=0)
    CN = VN - WN
    UA = np.round(np.sum(CN/VN)/Kclass*100, decimals=1)
    return UA

def getWA(OUT, TAR):
    DB_size = OUT.shape[0]
    OUT = np.argmax(OUT, axis=1)
    TAR = np.argmax(TAR, axis=1)
    hits = np.sum(OUT == TAR)
    WA = np.round(hits/DB_size*100, decimals=1)
    return WA
 


