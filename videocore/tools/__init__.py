from videocore.tools.animator.cli import AnimatorTool
from videocore.tools.base import BaseTool
from videocore.tools.enhancer.cli import EnhancerTool
from videocore.tools.face_studio.cli import FaceStudioTool
from videocore.tools.joiner.cli import JoinerTool
from videocore.tools.ui import UITool

TOOLS = [
    JoinerTool(),
    EnhancerTool(),
    AnimatorTool(),
    FaceStudioTool(),
    UITool(),
]

__all__ = ["BaseTool", "TOOLS", "JoinerTool", "EnhancerTool", "AnimatorTool", "FaceStudioTool", "UITool"]



