import numpy as np


dlc = dict(
    dlblue="#0096ff",
    dlorange="#FF9300",
    dldarkred="#C00000",
    dlmagenta="#FF40FF",
    dlpurple="#7030A0",
)


def load_house_data():
    data = np.loadtxt("../data/raw/houses.txt", delimiter=",", skiprows=1)
    x = data[:, :4]
    y = data[:, 4]
    return x, y
