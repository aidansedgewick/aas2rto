import sncosmo


def init_sncosmo_bandpasses():
    for b in "g r i".split():
        sncosmo.bandpasses.get_bandpass(f"ztf::{b}")
    for b in "u g r i z y".split():
        sncosmo.bandpasses.get_bandpass(f"lsst{b}")
    for b in "open g r i z y w".split():
        sncosmo.bandpasses.get_bandpass(f"ps1::{b}")
    for b in "c o".split():
        sncosmo.bandpasses.get_bandpass(f"atlas{b}")
    for b in "j h ks".split():
        sncosmo.bandpasses.get_bandpass(f"2mass{b}")
    for b in "b u uvm2 uvw1 uvw2 v white".split():
        sncosmo.bandpasses.get_bandpass(f"uvot::{b}")


if __name__ == "__main__":
    init_sncosmo_bandpasses()
