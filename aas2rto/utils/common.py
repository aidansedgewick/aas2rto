from __future__ import annotations

from astroplan import Observer


def chunk_list(l, chunk_size=100):
    for ii in range(0, len(l), chunk_size):
        yield l[ii : ii + chunk_size]


def get_observatory_name(observatory: Observer | str | None):
    if isinstance(observatory, str):
        return observatory
    return observatory.name


def format_link_as_html(link: str, text: str = "", prefix="https://"):
    text = text or link
    return f"<a href='{prefix}{link}'>{text}</a>"


def format_link_as_markdown(link: str, text: str = "", prefix="https://"):
    if text:
        return f"<{prefix}{link}|{text}>"
    return f"{prefix}{link}"
