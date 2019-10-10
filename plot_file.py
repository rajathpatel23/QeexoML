import matplotlib.pyplot as plt
import pywt
import numpy as np
import scipy.io.wavfile

if __name__ == '__main__':
    # location = "../data/train/hand-20140216_030347/20140216_025146693-knuckle/"
    location = "../data/train/hand-20140216_030347/20140216_025508803-pad/"
    data = scipy.io.wavfile.read(location + 'accel.wav')
    a = data[1].astype(float)
    scale = np.arange(1, 53)
    dt = 1 / 4000
    frequencies = pywt.scale2frequency('morl', scale) / dt
    print(frequencies)
    cwtmatr, freq = pywt.cwt(a, frequencies, 'morl')
    print(freq)
    cwtmatr_ = np.abs(cwtmatr)
    cwtmatr_ = np.divide(cwtmatr_ - np.min(cwtmatr_), (np.max(cwtmatr_) - np.min(cwtmatr_)))
    cwtmatr_ = np.flipud(cwtmatr_)
    print(cwtmatr_.shape)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9),
                                        tight_layout=True)
    t = np.arange(0, 256)
    print(t)
    im1 = ax1.plot(t, a)
    im2 = ax2.imshow(cwtmatr_, aspect='auto', interpolation="nearest", cmap='jet',
                     extent=[0,256, min(frequencies), max(frequencies)]

    ax1.set_title("Acclerometer vibration signal")
    ax1.set_xlabel("Time (msec)")
    ax1.set_ylabel("Amplitude")
    ax2.set_title("CWT Transform")
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (mins)')
    plt.show()