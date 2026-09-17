from portfolio_recsys.hooks import RunLoggingHooks
from kedro.config import OmegaConfigLoader


HOOKS = (RunLoggingHooks(),)

CONFIG_LOADER_CLASS = OmegaConfigLoader
