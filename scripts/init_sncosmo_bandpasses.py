# import requests
# import urllib
from pathlib import Path

try:
    import sncosmo
except ModuleNotFoundError as e:
    msg = (
        "`sncosmo` not imported properly. try:"
        "\n    \033[33;1mpython3 -m pip install sncosmo\033[0m"
    )
    raise ModuleNotFoundError(msg)

SVO_URL = "https://svo2.cab.inta-csic.es/svo/theory/fps/getdata.php?format=ascii"


def init_sncosmo_bandpasses():

    for b in "g r i".split():
        sncosmo.bandpasses.get_bandpass(f"ztf::{b}")
    for b in "u g r i z y".split():
        sncosmo.bandpasses.get_bandpass(f"lsst{b}")
    for b in "g r i z y w".split():
        sncosmo.bandpasses.get_bandpass(f"ps1::{b}")
    # for b in "c o".split():
    #    sncosmo.bandpasses.get_bandpass(f"atlas{b}") # ATLAS fails in GH actions
    # for b in "j h ks".split():
    #    sncosmo.bandpasses.get_bandpass(f"2mass{b}") # ATLAS fails in GH actions
    for b in "b u uvm2 uvw1 uvw2 v white".split():
        sncosmo.bandpasses.get_bandpass(f"uvot::{b}")


if __name__ == "__main__":
    init_sncosmo_bandpasses()
