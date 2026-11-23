"""过滤试跑引擎：在原触发规则之上，用 8 帧分数模板拦截反复误触发的背景声。

试跑规则（与核对页"过滤试跑"面板、README 一致）：
  - 从可触发状态（初始冷却剩余 r=0）按原规则回放：未处于冷却且分数达到阈值即触发，
    触发后接下来 cooldown 帧进入冷却。
  - 达阈且未冷却的帧，若帧号 >= 7 且当前已有模板，则取当前帧与前 7 帧共 8 个分数，
    按模板加入顺序逐条计算"逐项绝对差之和"（L1 距离）：
      * 第一条距离 <= tolerance 的模板命中 -> 拦下该帧，且【不启动冷却】；
      * 命中后不再与后续模板比较（已生效模板不重算距离）；
      * 全部模板都不命中 -> 按原规则触发并启动冷却。
  - 帧号 < 7（凑不齐 8 帧）或尚无模板时，一律按原规则触发。
  - 单遍线性回放，每帧只处理一次，已比过的帧不重算距离。
  - 真口令帧（protected）必须全部在触发帧里；漏掉即判不通过，调用方须保留
    原模板与原回放结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .engine import MAX_COOLDOWN, MAX_FRAMES, VerifyError

TEMPLATE_WINDOW = 8  # 模板长度：当前帧 + 前 7 帧
MAX_TEMPLATES = 8  # 模板最多累计 8 条


def validate_trial_inputs(
    scores: list[int],
    threshold: int,
    cooldown: int,
    tolerance: int,
    protected: list[int],
) -> int:
    """校验试跑输入，合法时返回帧数 n；不合法抛 VerifyError（中文可直接展示）。"""
    if not isinstance(scores, list) or not scores:
        raise VerifyError("分数序列不能为空。")
    n = len(scores)
    if n > MAX_FRAMES:
        raise VerifyError(f"分数帧数最多 {MAX_FRAMES}，当前 {n} 帧。")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in scores):
        raise VerifyError("分数必须全部为整数。")
    if not isinstance(threshold, int) or isinstance(threshold, bool):
        raise VerifyError("阈值必须为整数。")
    if not isinstance(cooldown, int) or isinstance(cooldown, bool):
        raise VerifyError("冷却帧数必须为整数。")
    if not 0 <= cooldown <= MAX_COOLDOWN:
        raise VerifyError(f"冷却帧数必须在 0～{MAX_COOLDOWN} 之间。")
    if not isinstance(tolerance, int) or isinstance(tolerance, bool):
        raise VerifyError("容差必须为非负整数。")
    if tolerance < 0:
        raise VerifyError("容差必须为非负整数。")
    if not isinstance(protected, list) or not protected:
        raise VerifyError("至少提供一个要保住的真口令帧。")
    if any(not isinstance(f, int) or isinstance(f, bool) for f in protected):
        raise VerifyError("真口令帧号必须全部为整数。")
    if any(not 0 <= f < n for f in protected):
        raise VerifyError(f"真口令帧号必须在 0..{n - 1} 之间。")
    if len(set(protected)) != len(protected):
        raise VerifyError("真口令帧号存在重复。")
    return n


def validate_candidate(
    frame: int,
    n: int,
    current_triggers: set[int],
    protected_set: set[int],
    used_frames: set[int],
    template_count: int,
) -> None:
    """校验工程师挑选的候选模板帧。"""
    if template_count >= MAX_TEMPLATES:
        raise VerifyError(f"模板最多累计 {MAX_TEMPLATES} 条，已达上限。")
    if not isinstance(frame, int) or isinstance(frame, bool):
        raise VerifyError("模板帧号必须为整数。")
    if frame < TEMPLATE_WINDOW - 1 or frame >= n:
        raise VerifyError(
            f"模板帧必须有完整的前 7 帧，帧号需在 {TEMPLATE_WINDOW - 1}..{n - 1} 之间。"
        )
    if frame in protected_set:
        raise VerifyError(f"帧 {frame} 是要保住的真口令帧，不能作为背景声模板。")
    if frame in used_frames:
        raise VerifyError(f"帧 {frame} 已经作为模板加入，不能重复选择。")
    if frame not in current_triggers:
        raise VerifyError(f"帧 {frame} 不是当前回放的触发帧，不能选为模板。")


@dataclass
class ReplayResult:
    triggers: list[int] = field(default_factory=list)
    blocked: list[dict] = field(default_factory=list)
    missed_protected: list[int] = field(default_factory=list)
    distance_evaluations: int = 0  # 实际计算过的（帧, 模板）距离条数
    compared_frames: int = 0  # 真正进入模板比对的帧数
    passed: bool = True


def replay(
    scores: list[int],
    threshold: int,
    cooldown: int,
    tolerance: int,
    protected: list[int],
    templates: list[list[int]],
) -> ReplayResult:
    """从可触发状态（r=0）按"原规则 + 模板拦截"单遍回放。

    纯函数：同样的模板集合无论分多少次加入，最终回放结果一致。
    """
    n = len(scores)
    tpls = [tuple(t) for t in templates]
    k = len(tpls)
    triggers: list[int] = []
    blocked: list[dict] = []
    remaining = 0  # 当前冷却剩余帧数；r=0 开始即可触发
    evals = 0
    compared = 0
    tol = tolerance

    for i in range(n):
        if remaining > 0:
            remaining -= 1
            continue
        if scores[i] < threshold:
            continue
        # 达阈且未冷却
        if k == 0 or i < TEMPLATE_WINDOW - 1:
            # 无模板，或凑不齐 8 帧：按原规则触发
            triggers.append(i)
            remaining = cooldown
            continue

        compared += 1
        window = scores[i - TEMPLATE_WINDOW + 1 : i + 1]
        hit_index = -1
        hit_distance = 0
        for ti, tpl in enumerate(tpls):
            evals += 1
            dist = 0
            for a, b in zip(window, tpl):
                dist += abs(a - b)
                if dist > tol:  # 提前剪枝：已超容差，本条模板必不命中
                    break
            if dist <= tol:
                hit_index = ti
                hit_distance = dist
                break  # 已生效模板命中即止，后续模板不重算距离

        if hit_index >= 0:
            blocked.append(
                {
                    "frame": i,
                    "score": scores[i],
                    "template_index": hit_index,
                    "distance": hit_distance,
                    "window": list(window),
                }
            )
            # 拦截帧不启动冷却：remaining 保持 0
        else:
            triggers.append(i)
            remaining = cooldown

    trigger_set = set(triggers)
    missed = sorted(f for f in protected if f not in trigger_set)
    return ReplayResult(
        triggers=triggers,
        blocked=blocked,
        missed_protected=missed,
        distance_evaluations=evals,
        compared_frames=compared,
        passed=not missed,
    )
