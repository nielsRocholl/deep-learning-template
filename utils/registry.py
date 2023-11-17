import inspect
from functools import partial
from utils import config

class Registry:
    """
    A registry to map strings to classes.
    Registered object could be built from registry.

    :param name: Registry name.
    :param build_func: Build function to construct instance from Registry.
    :param parent: Parent registry.
    :param scope: The scope of registry.
    """

    def __init__(self, name: str, build_func=None, parent=None, scope=None):
        self._name = name
        self._module_dict = dict()
        self._children = dict()
        self._scope = self.infer_scope() if scope is None else scope

        self.build_func = build_func or (parent.build_func if parent else build_from_cfg)
        self.parent = parent
        if parent:
            assert isinstance(parent, Registry)
            parent._add_children(self)

    def __len__(self) -> int:
        return len(self._module_dict)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self._name}, items={self._module_dict})'

    @staticmethod
    def infer_scope() -> str:
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
        return filename.split('.')[0]

    @staticmethod
    def split_scope_key(key: str) -> tuple:
        split_index = key.find('.')
        return (key[:split_index], key[split_index + 1:]) if split_index != -1 else (None, key)

    @property
    def name(self) -> str:
        return self._name

    @property
    def scope(self) -> str:
        return self._scope

    @property
    def module_dict(self) -> dict:
        return self._module_dict

    @property
    def children(self) -> dict:
        return self._children

    def get(self, key: str):
        scope, real_key = self.split_scope_key(key)
        if scope in {None, self._scope} and real_key in self._module_dict:
            return self._module_dict[real_key]
        return self._children.get(scope, self._get_root().get(key))

    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)

    def _add_children(self, registry):
        assert isinstance(registry, Registry)
        assert registry.scope and registry.scope not in self.children
        self.children[registry.scope] = registry

    def _register_module(self, module_class, module_name=None, force=False):
        assert inspect.isclass(module_class)
        module_name = module_name or module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError(f'{module_name} is already registered in {self.name}')
        self._module_dict[module_name] = module_class

    def register_module(self, name=None, force=False, module=None):
        if module:
            self._register_module(module_class=module, module_name=name, force=force)
            return module
        return partial(self._register_module, module_name=name, force=force)


def build_from_cfg(cfg: dict, registry: Registry, default_args: dict = None):
    assert isinstance(cfg, dict) and 'NAME' in cfg
    assert isinstance(registry, Registry)
    default_args = default_args or {}
    cfg = config.merge_new_config(cfg, default_args)
    obj_type = cfg.get('NAME')
    obj_cls = registry.get(obj_type) if isinstance(obj_type, str) else obj_type
    assert inspect.isclass(obj_cls)
    try:
        return obj_cls(cfg)
    except Exception as e:
        raise type(e)(f'{obj_cls.__name__}: {e}')
