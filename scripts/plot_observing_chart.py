from argparse import ArgumentParser

import pandas as pd

import matplotlib.pyplot as plt

from astropy.coordinates import EarthLocation
from astropy.time import Time

from astroplan import Observer

from dk154_targets.target import Target, plot_observing_chart

parser = ArgumentParser()
parser.add_argument("--coord", default=None)
parser.add_argument("--name", default="target")
parser.add_argument("--file", default=None)
parser.add_argument("--site", default="lasilla")
args = parser.parse_args()

target_list = []

if args.coord:
    if "," not in args.coord:
        print("provide coord as comma separated")
    ra_str, dec_str = args.coord.split(",")
    ra = float(ra_str.strip())
    dec = float(dec_str.strip())

    target_list = [Target(args.name, ra=ra, dec=dec)]

if args.file:
    data = pd.read_csv(args.file)
    target_list = []
    for ii, row in data.iterrows():
        target_list.append(Target(row.objectId, ra=row.ra, dec=row.dec))


site = args.site

location = EarthLocation.of_site(site)
observer = Observer(location=location, name=site)

t_ref = Time.now()

fig_ax = plot_observing_chart(
    observer, target_list=target_list, t_ref=t_ref, warn=False
)

plt.show()
