from logging import getLogger

try:
    import dustmaps
    from dustmaps import sfd
except ModuleNotFoundError as e:
    msg = "`dusmaps` not imported properly. try:\n    \033[33;1mpython3 -m pip install dustmaps\033[0m"
    raise ModuleNotFoundError(msg)


def init_sfd_dustmaps():
    logger.info("calling dustmaps.sfd.fetch()")
    sfd.fetch()


logger = getLogger(__name__.split(".")[-1])

if __name__ == "__main__":
    init_sfd_dustmaps()
