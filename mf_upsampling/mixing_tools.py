import numpy as np
import scipy.signal as sig

# From a wideband waveform, shift a band of selected frequencies towards the center
#
# This is to be done in two ways:
# 1. shift a single sided band towards the baseband. the center frequency represents the beginning of the frequency band
# 2. shift a double sided band. the center frequency represents the middle of the frequency band
#

# data import functions for USRP binary files
# (loading functions are tested - May 24)
def load_IQBinary_float32(file):
    # I and Q are both 32 bit floats
    return np.fromfile(file, dtype=np.complex64)


def load_IQBinary_float64(file):
    # I and Q are both 64 bit floats
    return np.fromfile(file, dtype=np.complex128)


def load_IQBinary_int8(file):
    # I and Q are both 8 bit ints
    x = np.fromfile(file, dtype=np.int8)
    x = x[0::2] + 1j*x[1::2]
    return x.astype('complex64')


def load_IQBinary_int16(file):
    # I and Q are both 16 bit ints
    x = np.fromfile(file, dtype=np.int16)
    x = x[0::2] + 1j*x[1::2]
    return x.astype('complex64')


# FFT frequency bucket indexing functions
# (tested for odd/even sized time series and on positive and negative frequencies- May 25)

# Note: rounding is performed to eliminate problems from floating point math adding any errors before conversion to integers. Rounding precision of 0.001 was selected arbitrarily but seemed reasonable.

# f is frequency to find index for
# n is number of samples in fft
# Ts is sampling time
# fc is center frequency (if the requested frequency is not baseband)

# positive frequency to lower FFT bucket:
#   given a positive freq, find the indice of the closest FFT bucket <= requested freq
def fpos2indFloor(f, n, Ts, fc=0, fout=False):
    ind = int(np.floor(np.round(n*(f-fc)*Ts, 3)))
    if fout:
        freq = fc + (ind/n)*(1/Ts)
        return ind, freq
    else:
        return ind


# positive frequency to upper FFT bucket:
#   given a positive freq, find the indice of the closest FFT bucket >= requested freq
def fpos2indCeil(f, n, Ts, fc=0, fout=False):
    ind = int(np.ceil(np.round(n*(f-fc)*Ts, 3)))
    if fout:
        freq = fc + (ind/n)*(1/Ts)
        return ind, freq
    else:
        return ind


# negative frequency to lower FFT bucket:
#   given a negative freq, find the indice of the closest FFT bucket <= requested freq
def fneg2indFloor(f, n, Ts, fc=0, fout=False):
    indOff = np.floor(np.round(n*(f+fc)*Ts, 3))
    ind = int(n + indOff)
    if fout:
        freq = fc + (indOff/n)*(1/Ts)
        return ind, freq
    else:
        return ind


# negative frequency to upper FFT bucket:
#   given a negative freq, find the indice of the closest FFT bucket >= requested freq
def fneg2indCeil(f, n, Ts, fc=0, fout=False):
    indOff = np.ceil(np.round(n*(f+fc)*Ts, 3))
    ind = int(n + indOff)
    if fout:
        freq = fc + (indOff/n)*(1/Ts)
        return ind, freq
    else:
        return ind


# given a set of positive frequency indices, get their corresponding complements in the negative frequency spectrum
def compFreq(n, indStart, indEnd):
    compStart = int(n - 1) - int(indStart)
    compEnd = int(n -1) - int(indEnd)
    return compStart, compEnd


# convenience function that gets the new sampling time after we've discarded some elements from the FFT series. Useful for finding new rate after mixing
def resampleTime(Ts, nfft, nfftNew):
    return Ts*nfft/nfftNew


# get the frequency bins for a frequency shifted signal.
# Useful in complement with mixing functions
def fc_fftfreq(n, Ts, fc):
    posLast = int(np.floor((n-1)/2))
    negFirst = posLast + 1
    print("posLast:", posLast)
    print("negFirst:", negFirst)
    freq = np.fft.fftfreq(n, Ts)
    freq[0:posLast+1] = freq[0:posLast+1] + fc
    freq[negFirst:] = freq[negFirst:] - fc
    return freq

# signal mixing functions

# these operate on frequency data. Another function is responsible for
# converting time series data into frequency data
# indStart is included. indEnd is not
def fft_basebandShift(fftdata, indStart, indEnd):
    n = fftdata.size
    if 2*(indEnd - indStart) > n:
        raise ValueError("requested bandwidth is larger than bandwidth of data")
    # fbb = np.zeros(2*nBuckets)
    # fbb[0:nBuckets] = fftdata[indStart:(indStart+nBuckets)]
    # fbb[nBuckets:] = fftdata[(n-(indStart+nBuckets)):(n-indStart)]
    fbb = np.concatenate((fftdata[indStart:indEnd],
                          fftdata[(n-indEnd):(n-indStart)]))
    return fbb


# given a data series recorded at sampling intervals Ts and center frequency fcData,
# return a new data series with only data in a certain bandwidth around or after fcSignal
def basebandShift(data, Ts, bwSignal,
                    fcSignal=0.0, fcData=0.0, signalType='ss'):
    n = data.size
    fdata = np.fft.fft(data)
    if signalType == 'ss':
        # signal is single sided
        fStart = fcSignal
        fEnd = fcSignal + bwSignal
    elif signalType == 'ds':
        # signal is double sided
        fStart = fcSignal - bwSignal/2
        fEnd = fcSignal + bwSignal/2
        
    # TODO: check if the frequencies lie in an acceptable range
    fmax = fcData + (n/2 -1)/(Ts*n) if n%2 == 0 else fcData + (n-1)/(2*Ts*n)
    if fStart < fcData or fStart > fmax or fEnd < fcData or fEnd > fmax:
        raise ValueError("requested bandwidth is out of supported range")
    
    indStart, f0 = fpos2indFloor(fStart, n, Ts, fc=fcData, fout=True)
    indEnd = fpos2indCeil(fEnd, n, Ts, fc=fcData, fout=False)
    fShifted = fft_basebandShift(fdata, indStart, indEnd+1)
    dataShifted = np.fft.ifft(fShifted)
    TsShifted = resampleTime(Ts, n, dataShifted.size)
    return dataShifted, f0, TsShifted


# helper function to generate a quadrature chirp
# t is a time series of the instances when we want to generate samples
def IQChirp(t, f0, t1, f1, method='linear', phi=0, vertex_zero=True):
    chirpI = sig.chirp(t, f0, t1, f1,
                    method=method,
                    phi=phi,
                    vertex_zero=vertex_zero)
    chirpQ = sig.chirp(t, f0, t1, f1,
                    method=method,
                    phi=phi-90.0,
                    vertex_zero=vertex_zero)
    return (chirpI + 1j*chirpQ)    

# template of a linear chirp
# Ts: sampling time
# fc: center frequency of chirp
# bw: bandwidth of chirp
# TChirp: chirp length
def chirpTemplate(Ts, fc, bw, TChirp, direction='up'):
    t = np.arange(0, TChirp, Ts)
    if direction == 'up':
        fStart = fc - bw/2
        fEnd = fc + bw/2
    elif direction == 'down':
        fStart = fc + bw/2
        fEnd = fc - bw/2
    chirp = IQChirp(t, fStart, TChirp, fEnd)
    return chirp


# generic matched filter function
# (tested on real and complex series of odd and even sizes - May 27)
def matchedFilter(data, template, mode='full'):
    m = np.flipud(np.conj(template))
    return np.convolve(data, m, mode=mode)


# return the output of a data series 
def chirpMF(data, Ts, fcData, fcChirp, bwChirp, TChirp):
    raise NotImplementedError
