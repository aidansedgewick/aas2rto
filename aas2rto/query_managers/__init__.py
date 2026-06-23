from aas2rto.query_managers.registry import (
    qm_registry,
    QueryManagerRegistry,
)  # First actual module import

from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.query_managers.primary import PrimaryQueryManager

# QMRegistry happens here!
# TODO: consider defining _autoregister() with importlib/pkgutils

# from aas2rto.query_managers.alerce import AlerceQueryManager
from aas2rto.query_managers.atlas import AtlasQueryManager
from aas2rto.query_managers.fink import FinkLSSTQueryManager, FinkZTFQueryManager

# from aas2rto.query_managers.lasair import LasairLSSTQueryManager, LasairZTFQueryManager
from aas2rto.query_managers.tns import TNSQueryManager
from aas2rto.query_managers.yse import YSEQueryManager

# from aas2rto.query_managers.sdss import SdssQueryManager
