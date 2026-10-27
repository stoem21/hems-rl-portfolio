import sys
import os

# Allow tests to import backend modules (main, hems_env, data_config, agent_loader)
# without installing the package. PYTHONPATH=hems-backend is set in CI; this
# file covers the case where tests are run locally from the project root.
_BACKEND = os.path.join(os.path.dirname(__file__), "..", "hems-backend")
if os.path.abspath(_BACKEND) not in [os.path.abspath(p) for p in sys.path]:
    sys.path.insert(0, os.path.abspath(_BACKEND))
