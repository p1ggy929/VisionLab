"""既有 .pth 权重的 state_dict 键名兼容层。

UNet / UNetV2 / Attention-ResUNet 的结构移植自原项目 B，早期权重以
CamelCase 的层名（如 ``Down1``、``Translayer_1``、``W_g``）保存。为满足统一
命名规范，层名已改为 snake_case；加载旧权重时先经过本模块把键名映射到新名字，
避免"结构对但键名不匹配"导致既有权重无法加载。
"""

from __future__ import annotations

# 旧键名片段 -> 当前键名片段（按 state_dict 键的 "." 分段替换）
LEGACY_KEY_ALIASES = {
    "Down1": "down1",
    "Down2": "down2",
    "Down3": "down3",
    "Down4": "down4",
    "Up1": "up1",
    "Up2": "up2",
    "Up3": "up3",
    "Up4": "up4",
    "Translayer_1": "translayer_1",
    "Translayer_2": "translayer_2",
    "Translayer_3": "translayer_3",
    "Translayer_4": "translayer_4",
    "W_g": "w_g",
    "W_x": "w_x",
}


def remap_legacy_state_dict(state: dict) -> dict:
    """把旧版 CamelCase 键名转换为当前的 snake_case 键名。

    已经是新命名的权重不受影响（映射表未命中时原样返回），因此新旧权重都能加载。
    """
    if not isinstance(state, dict):
        return state
    remapped: dict = {}
    for key, value in state.items():
        parts = [LEGACY_KEY_ALIASES.get(segment, segment) for segment in str(key).split(".")]
        remapped[".".join(parts)] = value
    return remapped


__all__ = ["LEGACY_KEY_ALIASES", "remap_legacy_state_dict"]
