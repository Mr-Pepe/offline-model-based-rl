# Based on https://spinningup.openai.com

import json


def convert_json(obj: object):
    """Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj

    if isinstance(obj, dict):
        return {convert_json(k): convert_json(v) for k, v in obj.items()}

    if isinstance(obj, tuple):
        return (convert_json(x) for x in obj)

    if isinstance(obj, list):
        return [convert_json(x) for x in obj]

    if hasattr(obj, "__name__") and not "lambda" in obj.__name__:
        return convert_json(obj.__name__)

    if hasattr(obj, "__dict__") and obj.__dict__:
        obj_dict = {convert_json(k): convert_json(v) for k, v in obj.__dict__.items()}
        return {str(obj): obj_dict}

    return str(obj)


def is_json_serializable(obj: object):
    try:
        json.dumps(obj)
        return True
    except:  # pylint: disable=bare-except
        return False
