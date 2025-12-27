"""
保持 joyful 包可被轻量导入：
- 训练环境通常有 torch / sentence_transformers，可完整导入。
- 仅用于反序列化(pickle)时，可能没有 torch；此时也应能 import joyful 并找到 Sample 类，
  避免工具脚本在无 torch 环境下直接失败。
"""

# Sample：允许在无 torch 环境下被 import（Sample.py 已改为惰性加载 sbert/torch）
try:
    from .Sample import Sample  # noqa: F401
except Exception:
    Sample = None  # type: ignore

# 其余训练相关组件：按需导入，缺依赖时保持可用性
try:
    import joyful.utils  # noqa: F401
    from .Dataset import Dataset  # noqa: F401
    from .Coach import Coach  # noqa: F401
    from .model.JOYFUL import JOYFUL  # noqa: F401
    from .Optim import Optim  # noqa: F401
except Exception:
    pass
