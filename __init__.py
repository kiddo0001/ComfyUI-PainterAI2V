from .PainterAI2V import PainterAI2V
from .PainterAV2V import PainterAV2V
from comfy_api.latest import ComfyExtension, io
from typing import override

__version__ = "1.0.0"

class PainterExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PainterAI2V, PainterAV2V]

async def comfy_entrypoint():
    return PainterExtension()
