import inspect
import json
import jinja2
try:
    import replicate
except ImportError:
    replicate = None
from openai import OpenAI
from types import MethodType


registry = {}
NON_SERIALIZABLE_TYPES = [jinja2.Environment, MethodType, OpenAI]
if replicate is not None:
    NON_SERIALIZABLE_TYPES.append(replicate.Client)
NON_SERIALIZABLE_TYPES = tuple(NON_SERIALIZABLE_TYPES)


def register_class(target_class):
    registry[target_class.__name__] = target_class


class Meta(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        register_class(cls)
        return cls


class RegisteredSerializable(metaclass=Meta):
    def __init__(self, *args) -> None:
        self.args = args

    def to_serial_dict(self) -> dict:
        def serialize_value(v):
            if isinstance(v, set):
                return {"__set__": list(v)}
            elif isinstance(v, dict):
                return {k: serialize_value(vv) for k, vv in v.items()}
            elif isinstance(v, list):
                return [serialize_value(vv) for vv in v]
            elif isinstance(v, RegisteredSerializable):
                return v.to_serial_dict()
            else:
                return v

        serial_dict = {"class": self.__class__.__name__, "args": {}}
        for k, v in self.__dict__.items():
            if not callable(v) and not isinstance(v, NON_SERIALIZABLE_TYPES):
                serial_dict["args"][k] = serialize_value(v)
        return serial_dict

    @classmethod
    def from_dict(cls, data):
        def deserialize_value(value) -> dict:
            if isinstance(value, dict) and "class" in value:
                # debugging print line
                # print(f"Deserializing class: {value['class']}")
                return RegisteredSerializable.from_dict(value)
            elif isinstance(value, dict) and "__set__" in value:
                return set(value["__set__"])
            elif isinstance(value, dict):
                return {k: deserialize_value(vv) for k, vv in value.items()}
            elif isinstance(value, list):
                return [deserialize_value(vv) for vv in value]
            else:
                return value

        class_name = data["class"]
        target_class = registry[class_name]
        args = data["args"]
        init_args = inspect.getfullargspec(target_class.__init__).args[
            1:
        ]  # Exclude 'self'

        if len(init_args) == 0:
            instance = target_class()
        else:
            valid_args = {k: v for k, v in args.items() if k in init_args}
            instance = target_class(**valid_args)

        for key, value in args.items():
            # debugging print line
            # print(f"Setting attribute: {key}")
            setattr(instance, key, deserialize_value(value))
        return instance

    def serialize(self) -> str:
        return json.dumps(self.to_serial_dict())

    @staticmethod
    def deserialize(data) -> "RegisteredSerializable":
        # debugging print line
        # print(registry)
        params = json.loads(data)
        return RegisteredSerializable.from_dict(params)
