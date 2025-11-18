import sys

# this will break any attempt to import xformers which will prevent stability diffusion repo from trying to use it
if "--xformers" not in "".join(sys.argv):
    sys.modules["xformers"] = None

# Hack to fix a changed import in torchvision 0.17+, which otherwise breaks
# basicsr; see https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ImportError:
    try:
        import torchvision.transforms.functional as functional
        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass  # shrug...

# Fix for transformers 4.56.x: GenerationMixin import error
# This function should be called before pytorch_lightning imports transformers
def fix_transformers_generation_mixin():
    """Fix transformers.generation.GenerationMixin import issue for transformers 4.56.x"""
    try:
        import transformers
        if not hasattr(transformers, 'generation'):
            return
        
        generation_module = transformers.generation
        if hasattr(generation_module, 'GenerationMixin'):
            return
        
        # Try to import GenerationMixin from various possible locations
        try:
            from transformers.generation.utils import GenerationMixin as GenMixin
            generation_module.GenerationMixin = GenMixin
            return
        except (ImportError, AttributeError):
            pass
        
        try:
            from transformers.generation.configuration_utils import GenerationMixin as GenMixin
            generation_module.GenerationMixin = GenMixin
            return
        except (ImportError, AttributeError):
            pass
        
        # Try to find GenerationMixin in transformers.generation submodules
        try:
            import importlib
            import inspect
            gen_package = importlib.import_module('transformers.generation')
            for attr_name in dir(gen_package):
                attr = getattr(gen_package, attr_name, None)
                if inspect.isclass(attr) and attr_name == 'GenerationMixin':
                    generation_module.GenerationMixin = attr
                    return
        except Exception:
            pass
        
        # Last resort: create a dummy class to prevent import errors
        class GenerationMixin:
            """Dummy GenerationMixin for compatibility"""
            pass
        generation_module.GenerationMixin = GenerationMixin
    except Exception:
        pass
