import numpy as np
import matplotlib.pylab as mpl
import scipy.interpolate
import sys

""" Based loosely on http://andrewwalker.github.io/statefultransitions/post/gaussian-fields/ """

""" Downscales a gridded forecast to a higher time resolution

    Arguments:
        array (np.array): 3D array (X, Y, Time)
        n (int): How many new timesteps per exising timestep?
        rx (float): Spatial tuning parameter. Higher number means higher spatial correlation.
        rt (float): Temporal tuning parameter. Higher number means higher temporal correlation.
        a (float): Stdev of noise (in log wind speed space)

    Returns:
        array (np.array): 3D array (X, Y, Time)
"""
def downscale_field(array, n, rx, rt, a):
    X, Y, T = array.shape
    T2 = T * n

    # Linearly interpolate the forecast to new time dimension
    func = scipy.interpolate.interp1d(np.linspace(0, 1, T), array, axis=2)
    ret = func(np.linspace(0, 1.0, T2))

    noise = gaussian_random_field(X, Y, T2, rx, rt)

    """ Adjust the field by adding the noise. This is done in log windspeed space.
    """
    new_array = np.exp(np.log(ret) + noise * a)
    # new_array = noise

    return new_array


""" Creates a gaussian random field with given spatial/temporal characteristics and a stdev of 1

    Arguments:
        X (int): Number of gridpoints in X direction
        Y (int): Number of gridpoints in Y direction
        T (int): Number of timesteps
        rx (float): Spatial tuning parameter. Higher number means higher spatial correlation.
        rt (float): Temporal tuning parameter. Higher number means higher temporal correlation.

    Returns:
        array (np.array): 3D array (X, Y, Time)
"""
def gaussian_random_field(X, Y, T, rx, rt):
    # Create Gaussian noise field
    rand = np.random.normal(size = (X, Y, T))

    noise = np.fft.fftn(rand)
    amplitude = np.zeros([X, Y, T])
    for i, kx in enumerate(fftIndgen(X)):
        for j, ky in enumerate(fftIndgen(Y)):
            for t, kt in enumerate(fftIndgen(T)):
                amplitude[i, j, t] = Pk(kx, ky, kt, rx, rt)

    noise2 = np.fft.ifftn(noise * amplitude)

    # Normalize std to be 1
    noise2 = noise2 / np.std(noise2)
    return noise2.real


""" I think this calculates how much to weight different fourier components ... """
def Pk(kx, ky, kt, rx, rt):
    if (kx == 0 and ky == 0) or kt == 0:
        return 0.0
    spatial = np.sqrt(kx**2 + ky**2)**(-rx/2.0)
    temporal = (np.abs(kt))**(-rt/2.0)
    return spatial * temporal


""" Not sure what this does ... """
def fftIndgen(n):
    a = range(0, n/2+1)
    b = range(1, n/2)
    b.reverse()
    b = [-i for i in b]
    return a + b


np.random.seed(1000)

# Create a EC wind forecast with 5 timesteps of 6 hours
raw_forecast = 3 * np.ones([100,100,5])

# Downscale to 1 hour resolution
x = downscale_field(raw_forecast, n=6, rx=4, rt=2, a=0.1)

T = x.shape[2]
for i in range(T):
    mpl.subplot(int(np.sqrt(T)), T / int(np.sqrt(T))+1,i+1)
    mpl.pcolormesh(x[:, :, i], vmin=np.min(x[:]), vmax=np.max(x[:]), cmap="RdBu_r")
    mpl.gca().set_aspect(1)
    mpl.title("Timestep %d" % i)
mpl.show()

# Show a timeseries plot
mpl.plot(x[x.shape[0]/2, x.shape[1]/2, :])
mpl.title("Timeseries for one point")
mpl.xlabel("Time (h)")
mpl.ylabel("Wind speed (m/s)")
mpl.show()
