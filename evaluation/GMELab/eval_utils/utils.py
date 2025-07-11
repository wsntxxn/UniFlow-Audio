import dataclasses

from eval_utils.exceptions import ConfigurationError


def dataclass_from_dict(klass, d):
    try:
        fieldtypes = {f.name: f.type for f in dataclasses.fields(klass)}
        return klass(**{f: dataclass_from_dict(fieldtypes[f], d[f]) for f in d})
    except ConfigurationError as e:  # Invalid value
        raise e
    except Exception as _:
        return d  # Not a dataclass field
