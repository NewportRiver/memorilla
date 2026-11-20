__all__ = ['TrainingConfig', 'load_development', 'TrainingExample', 'rss_bytes', 'available_ram_bytes']


def __getattr__(name):
    if name == 'TrainingConfig':
        from .config import TrainingConfig
        return TrainingConfig
    if name in ('load_development', 'TrainingExample'):
        from .dataset import TrainingExample, load_development
        return {'load_development': load_development, 'TrainingExample': TrainingExample}[name]
    if name in ('rss_bytes', 'available_ram_bytes'):
        from .numpy_telemetry import available_ram_bytes, rss_bytes
        return {'rss_bytes': rss_bytes, 'available_ram_bytes': available_ram_bytes}[name]
    raise AttributeError(name)
