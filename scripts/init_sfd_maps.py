from argparse import ArgumentParser


def init_sfd_dustmaps(reset_config: bool = False):
    try:
        from dustmaps.config import config

    except ModuleNotFoundError as e:
        msg = (
            "`dustmaps` not imported properly. try:"
            "\n    \033[33;1mpython3 -m pip install dustmaps\033[0m"
        )
        raise ModuleNotFoundError(msg)

    if reset_config:
        print("reset config")
        config.reset()  # Stop dustmaps config warning in CI

    print("import sfd")
    from dustmaps import sfd

    print("calling dustmaps.sfd.fetch()")
    sfd.fetch()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--reset-config", default=False, action="store_true")
    args = parser.parse_args()

    init_sfd_dustmaps(reset_config=args.reset_config)
