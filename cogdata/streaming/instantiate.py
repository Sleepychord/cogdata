from omegaconf import OmegaConf

def instantiate_from_yaml(config_path, variables={}):
    OmegaConf.register_new_resolver("variables", lambda x: variables.get(x, x))
    try:
        config = OmegaConf.load(config_path)
        OmegaConf.resolve(config)
        x = _recursive_instantiate_from_yaml(config)
    finally:
        # remove resolver
        OmegaConf.clear_resolvers()
    return x

def _recursive_instantiate_from_yaml(config):
    from omegaconf import OmegaConf
    if OmegaConf.is_dict(config):
        conf = {}
        for k, v in config.items():
            conf[k] = _recursive_instantiate_from_yaml(v)
    elif OmegaConf.is_list(config):
        conf = []
        for i, v in enumerate(config):
            conf.append(_recursive_instantiate_from_yaml(v))
    else:
        return config
    
    # already recursive solved children
    if "include" in conf:
        assert "target" not in conf and "params" not in conf, "included {conf} should not have target or params"
        sub_config = OmegaConf.load(config["include"])
        OmegaConf.resolve(sub_config)
        x = _recursive_instantiate_from_yaml(sub_config)
        # override with current config, e.g. percent
        for k, v in conf.items():
            if k != "include":
                setattr(x, k, v)
        return x
    
    if "target" in conf:
        # instantiate a obj
        if "params" in conf:
            if conf['params'] is None:
                x = get_obj_from_str(conf["target"])()
            else:
                x = get_obj_from_str(conf["target"])(**conf.get("params", dict()))
        else:
            x = get_obj_from_str(conf["target"])
        for k, v in conf.items():
            if k != "target" and k != "params":
                setattr(x, k, v)
        return x
    return conf


def get_obj_from_str(string, reload=True):
    import importlib
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)