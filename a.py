import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.basemap import Basemap
from ast import literal_eval


plt.rcParams['animation.ffmpeg_path'] = "C:\FFmpeg\\bin\\ffmpeg.exe"


def get_data(table, rownum, title):
    data = pd.DataFrame(table.loc[rownum])
    data.columns = {title}
    return data


def augment(xold, yold, numsteps):
    xnew = []
    ynew = []
    for i in range(len(xold)-1):
        difX = xold[i+1]-xold[i]
        stepsX = difX/numsteps
        difY = yold[i+1]-yold[i]
        stepsY = difY/numsteps
        for s in range(numsteps):
            xnew = np.append(xnew, xold[i]+s*stepsX)
            ynew = np.append(ynew, yold[i]+s*stepsY)
    return xnew, ynew


def smoothListGaussian(listin, strippedXs=False, degree=5):
    window = degree*2-1
    weight = np.array([1.0]*window)
    weightGauss = []
    for i in range(window):
        i = i-degree+1
        frac = i/float(window)
        gauss = 1/(np.exp((4*(frac))**2))
        weightGauss.append(gauss)
    weight = np.array(weightGauss)*weight
    smoothed = [0.0]*(len(listin)-window)
    for i in range(len(smoothed)):
        smoothed[i] = sum(np.array(listin[i:i+window])*weight)/sum(weight)
    return smoothed


csv = pd.read_csv('./dataone.csv')


x = list(csv["Year"].astype('category').value_counts().index)
x.sort()
csv.sort_values(by=['Year'], inplace=True)
y = list(csv["Year"].value_counts().sort_index())
XN, YN = augment(x, y, 10)
XN = smoothListGaussian(XN)
YN = smoothListGaussian(YN)
# augmented =
print(y)

print(x)

title = 'Eye Diseases'

df = pd.DataFrame(YN, XN)
#XN,YN = augment(x,y,10)
#augmented = pd.DataFrame(YN,XN)
df.columns = {title}

Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)


# fig = plt.figure(figsize=(10, 6))
# plt.xlim(2005, 2016)
# plt.ylim(np.min(df)[0], np.max(df)[0])
# plt.xlabel('Year', fontsize=20)
# plt.ylabel(title, fontsize=20)
# plt.title('Eye Diseases per Year', fontsize=20)


def animate(i):
    data = df.iloc[:int(i+1)]  # select data range
    p = sns.lineplot(x=data.index, y=data[title], data=data, color="r")
    p.tick_params(labelsize=17)
    plt.setp(p.lines, linewidth=7)


sns.set(rc={'axes.facecolor': 'lightgrey', 'figure.facecolor': 'lightgrey',
            'figure.edgecolor': 'black', 'axes.grid': False})
# ani = matplotlib.animation.FuncAnimation(fig, animate, frames=150, repeat=True)
# ani.save('test.mp4', writer=writer)
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution=None,
            width=8E6, height=8E6,
            lat_0=45, lon_0=-100,)
m.etopo(scale=0.5, alpha=0.5)

for loc in csv["GeoLocation"].unique():
    weight = csv["GeoLocation"].value_counts()[loc]//250
    lat, long = literal_eval(loc)
    x, y = m(long, lat)
    plt.plot(x, y, 'ok', markersize=weight)

plt.show()
