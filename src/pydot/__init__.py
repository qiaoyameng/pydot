# SPDX-FileCopyrightText: 2024 pydot contributors
#
# SPDX-License-Identifier: MIT

"""An interface to Graphviz."""

import logging

__author__ = "Ero Carrera"
__version__ = "4.0.2.dev0"
__license__ = "MIT"


_logger = logging.getLogger(__name__)
_logger.debug("pydot initializing")
_logger.debug("pydot %s", __version__)


from pydot import xdot  # noqa: E402, F401
from pydot.classes import FrozenDict  # noqa: F401, E402
from pydot.core import *  # noqa: F403, E402
from pydot.exceptions import *  # noqa: E402, F403
from pydot.xdot import *  # noqa: E402, F403
