"""视界智研算法实验平台（VisionLab-D455）。

技术栈沿用两个原项目：
  Web      : Flask + Flask-SocketIO + eventlet + Jinja2 + 原生 JS + Plotly   (项目 A)
  算法     : PyTorch + Ultralytics + OpenCV + NumPy / Pandas / SciPy          (项目 A / B)
  相机     : pyrealsense2 (Intel RealSense D455)                              (项目 A / B)
  报告     : ReportLab                                                        (项目 A)
  元数据库 : Python 标准库 sqlite3（原项目无数据库，新增且零依赖）
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
