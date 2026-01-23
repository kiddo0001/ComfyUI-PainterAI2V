# 修改 __init__.py 内容
from .PainterAI2V import PainterAI2VExtension

__version__ = "1.0.0"

async def comfy_entrypoint():
    return PainterAI2VExtension()
