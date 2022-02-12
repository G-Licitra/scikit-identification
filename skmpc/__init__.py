# import scikit-mpc objects
from skmpc.integrator import *
from skmpc.models import *

# Current version
__version__ = "0.1.0"

# Warn if a newer version of Pingouin is available
from outdated import warn_if_outdated

warn_if_outdated("scikit-mpc", __version__)

# load default options
# set_default_options()
