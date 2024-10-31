

def param_to_buffer(module):
    """Turns all parameters of a module into buffers."""
    modules = module.modules()
    module = next(modules)
    for name, param in list(module.named_parameters(recurse=False)):
        if param.requires_grad:
            continue
        # Unregister parameter
        delattr(module, name)
        module.register_buffer(name, param)
    for module in modules:
        param_to_buffer(module)
