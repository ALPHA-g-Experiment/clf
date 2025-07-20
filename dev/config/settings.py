from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="CLF_TRAIN",
    settings_file="default.toml",
)
