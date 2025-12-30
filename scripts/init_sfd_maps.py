def init_sfd_dustmaps():
    try:
        import dustmaps
        from dustmaps import sfd
    except ModuleNotFoundError as e:
        msg = (
            "`dusmaps` not imported properly. try:"
            "\n    \033[33;1mpython3 -m pip install dustmaps\033[0m"
        )
        raise ModuleNotFoundError(msg)

    print("calling dustmaps.sfd.fetch()")
    sfd.fetch()

if __name__ == "__main__":
    init_sfd_dustmaps()