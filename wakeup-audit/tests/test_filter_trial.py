"""过滤试跑引擎测试：规则用例 + 对暴力参考实现的随机对拍 + 输入校验 + 性能。

运行：python -I -m pytest tests/test_filter_trial.py -q
"""

from __future__ import annotations

import random
import time

import pytest

from app.engine import MAX_COOLDOWN, MAX_FRAMES
from app.filter_trial import (
    MAX_TEMPLATES,
    TEMPLATE_WINDOW,
    VerifyError,
    replay,
    validate_candidate,
    validate_trial_inputs,
)


# ---------------------------------------------------------------- 暴力参考实现
def brute_replay(scores, threshold, cooldown, tolerance, templates):
    """逐帧暴力参考：与 replay 同样的规则，写法尽量直白、独立实现。"""
    n = len(scores)
    triggers: list[int] = []
    blocked: list[dict] = []
    remaining = 0
    evals = 0
    compared = 0
    for i in range(n):
        if remaining > 0:
            remaining -= 1
            continue
        if scores[i] < threshold:
            continue
        if not templates or i < TEMPLATE_WINDOW - 1:
            triggers.append(i)
            remaining = cooldown
            continue
        compared += 1
        window = scores[i - TEMPLATE_WINDOW + 1 : i + 1]
        hit = None
        for ti, tpl in enumerate(templates):
            evals += 1
            dist = sum(abs(a - b) for a, b in zip(window, tpl))
            if dist <= tolerance:
                hit = (ti, dist)
                break
        if hit is None:
            triggers.append(i)
            remaining = cooldown
        else:
            blocked.append(
                {"frame": i, "template_index": hit[0], "distance": hit[1]}
            )
    return triggers, blocked, evals, compared


def assert_replay_matches(scores, threshold, cooldown, tolerance, templates, protected):
    res = replay(scores, threshold, cooldown, tolerance, protected, templates)
    b_triggers, b_blocked, b_evals, b_compared = brute_replay(
        scores, threshold, cooldown, tolerance, templates
    )
    assert res.triggers == b_triggers
    assert [(b["frame"], b["template_index"], b["distance"]) for b in res.blocked] == [
        (b["frame"], b["template_index"], b["distance"]) for b in b_blocked
    ]
    assert res.distance_evaluations == b_evals
    assert res.compared_frames == b_compared
    return res


# ---------------------------------------------------------------- 规则用例
def test_base_replay_equals_original_rule():
    # 无模板时与原触发规则完全一致：r=0、达阈触发、触发后冷却 cooldown 帧
    scores = [8, 8, 7, 8, 7, 4, 3, 5, 6, 3, 8, 9]
    res = replay(scores, threshold=5, cooldown=2, tolerance=0, protected=[7, 10], templates=[])
    assert res.triggers == [0, 3, 7, 10]
    assert res.blocked == []
    assert res.passed is True
    assert res.missed_protected == []
    assert res.distance_evaluations == 0
    assert res.compared_frames == 0


def test_first_seven_frames_always_fire_when_crossing():
    # 帧 0..6 凑不齐 8 帧，即使给模板也按原规则触发
    scores = [9] * 7 + [0] * 3
    tpl = [1] * 8
    res = replay(scores, 5, 0, tolerance=10**9, protected=[0], templates=[tpl])
    assert res.triggers == [0, 1, 2, 3, 4, 5, 6]
    assert res.blocked == []
    assert res.compared_frames == 0


def test_template_blocks_without_cooldown_chain_effect():
    # 帧 7 被拦且不启冷却 -> 帧 8 达阈立刻顶上来触发（拦截的连锁效应）
    scores = [0] * 20
    scores[3:11] = [1, 2, 3, 4, 9, 9, 8, 7]  # 帧 7 达阈；帧 8、9 也达阈
    # 帧 10 达阈且形状不同，作为要保住的真口令
    scores[10] = 9
    tpl = scores[0:8]
    res = replay(scores, 5, 2, tolerance=0, protected=[10], templates=[tpl])
    assert [b["frame"] for b in res.blocked] == [7]
    assert res.blocked[0]["template_index"] == 0
    assert res.blocked[0]["distance"] == 0
    # 7 被拦不冷却 -> 8 触发 -> 9、10 冷却 -> 真口令 10 漏触发，判定不通过
    assert res.triggers == [8]
    assert res.passed is False
    assert res.missed_protected == [10]


def test_same_template_blocks_repeated_background_keeps_command():
    # 冷却 0：同形背景声在 10/20 两次出现，被同一模板拦下；异形真口令 30 保住；
    # 真口令之后帧 32 再次出现同形背景声，仍被同一模板拦下。
    n = 40
    scores = [0] * n
    shape = [5, 5, 5, 5, 5, 5, 5, 9]
    command = [9, 9, 9, 9, 9, 9, 9, 9]

    def place(end_frame: int, window: list[int]) -> None:
        start = end_frame - TEMPLATE_WINDOW + 1
        for off, v in enumerate(window):
            scores[start + off] = v

    place(10, shape)   # 背景 1：窗口帧 3..10
    place(20, shape)   # 背景 2：窗口帧 13..20
    place(30, command)  # 真口令：窗口帧 23..30，异形
    place(39, shape)   # 背景 3：窗口帧 32..39，与真口令窗口不重叠
    tpl = scores[3:11]
    res = replay(scores, 5, 0, tolerance=0, protected=[30], templates=[tpl])
    assert [b["frame"] for b in res.blocked] == [10, 20, 39]
    assert all(b["template_index"] == 0 for b in res.blocked)
    assert 30 in res.triggers
    assert res.passed is True


def test_tolerance_boundary_inclusive():
    # L1 距离恰好等于容差 -> 拦截（"不超过"含等号）
    scores = [0] * 16
    scores[8] = 9  # 帧 8 达阈
    tpl = scores[1:9]
    tpl[0] = 3  # 仅首元素差 3，其余相同 -> L1 = 3
    res_eq = replay(scores, 5, 0, tolerance=3, protected=[], templates=[tpl])
    assert [b["frame"] for b in res_eq.blocked] == [8]
    res_lt = replay(scores, 5, 0, tolerance=2, protected=[], templates=[tpl])
    assert res_lt.blocked == []
    assert res_lt.triggers == [8]


def test_first_matching_template_wins_and_no_recompute():
    # 两个模板都能命中帧 8 时，只取模板 0；统计的距离条数证明未与模板 1 重算
    scores = [0] * 16
    scores[8] = 9
    tpl0 = scores[1:9]
    tpl1 = list(tpl0)
    tpl1[7] = 8  # 与帧 8 窗口距离 1，容差 5 下也会命中
    res = replay(scores, 5, 0, tolerance=5, protected=[], templates=[tpl0, tpl1])
    assert [b["frame"] for b in res.blocked] == [8]
    assert res.blocked[0]["template_index"] == 0
    assert res.distance_evaluations == 1  # 命中即止，模板 1 不重算


def test_pure_function_template_set_order_independent_outcome():
    # 同一模板集合无论怎么分批加入，最终回放一致（引擎是纯函数）
    rng = random.Random(4242)
    scores = [rng.randint(0, 12) for _ in range(60)]
    tpls = [scores[i - 7 : i + 1] for i in (10, 25, 40)]
    one_shot = replay(scores, 8, 1, 3, [55], tpls)
    accumulated = replay(scores, 8, 1, 3, [55], tpls[:1])
    accumulated = replay(scores, 8, 1, 3, [55], tpls[:2])
    accumulated = replay(scores, 8, 1, 3, [55], tpls)
    assert accumulated.triggers == one_shot.triggers
    assert [b["frame"] for b in accumulated.blocked] == [
        b["frame"] for b in one_shot.blocked
    ]


def test_missed_protected_listed_sorted():
    scores = [0] * 30
    scores[7] = 9
    scores[20] = 9
    # 容差极大：两个达阈帧都被拦，两个真口令全漏
    tpl = scores[0:8]
    res = replay(scores, 5, 0, tolerance=10**9, protected=[20, 7], templates=[tpl])
    assert res.triggers == []
    assert res.missed_protected == [7, 20]
    assert res.passed is False


# ---------------------------------------------------------------- 随机对拍
def test_property_random_vs_bruteforce():
    rng = random.Random(20261123)
    for _ in range(400):
        n = rng.randint(8, 60)
        scores = [rng.randint(0, 15) for _ in range(n)]
        threshold = rng.randint(2, 16)
        cooldown = rng.randint(0, 6)
        tolerance = rng.randint(0, 20)
        k = rng.randint(0, 4)
        # 模板取自分数自身的合法 8 帧窗口，模拟"从触发帧取模板"
        tpls = [
            scores[rng.randint(TEMPLATE_WINDOW - 1, n - 1) - 7 :][:8]
            for _ in range(k)
        ]
        protected = sorted(rng.sample(range(n), rng.randint(1, 3)))
        res = assert_replay_matches(
            scores, threshold, cooldown, tolerance, tpls, protected
        )
        # 漏报集合必须等于"真口令帧不在触发帧里"
        assert res.missed_protected == sorted(
            f for f in protected if f not in set(res.triggers)
        )
        assert res.passed == (not res.missed_protected)
        # 拦截帧与触发帧互不相交，且拦截帧都达阈
        blocked_frames = {b["frame"] for b in res.blocked}
        assert blocked_frames.isdisjoint(set(res.triggers))
        assert all(scores[f] >= threshold for f in blocked_frames)


def test_property_eight_templates_full_capacity():
    rng = random.Random(99)
    scores = [rng.randint(0, 9) for _ in range(80)]
    tpls = [scores[i - 7 : i + 1] for i in range(7, 7 + MAX_TEMPLATES)]
    res = assert_replay_matches(scores, 5, 3, 4, tpls, [70, 71])
    # 每个进入比对的帧最多算满 8 条距离
    assert res.distance_evaluations <= res.compared_frames * MAX_TEMPLATES


# ---------------------------------------------------------------- 校验
def test_validate_trial_inputs():
    ok_scores = [1, 2, 3] * 4
    validate_trial_inputs(ok_scores, 5, 2, 0, [0])  # 不抛即通过
    with pytest.raises(VerifyError):
        validate_trial_inputs([], 5, 2, 0, [0])
    with pytest.raises(VerifyError):
        validate_trial_inputs([1] * (MAX_FRAMES + 1), 5, 2, 0, [0])
    with pytest.raises(VerifyError):
        validate_trial_inputs([1, 2.0, 3], 5, 2, 0, [0])  # 非整数分数
    with pytest.raises(VerifyError):
        validate_trial_inputs(ok_scores, 5, MAX_COOLDOWN + 1, 0, [0])
    with pytest.raises(VerifyError):
        validate_trial_inputs(ok_scores, 5, 2, -1, [0])  # 负容差
    with pytest.raises(VerifyError):
        validate_trial_inputs(ok_scores, 5, 2, 1.5, [0])  # 非整数容差
    with pytest.raises(VerifyError):
        validate_trial_inputs(ok_scores, 5, 2, 0, [])  # 没有真口令帧
    with pytest.raises(VerifyError):
        validate_trial_inputs(ok_scores, 5, 2, 0, [99])  # 越界
    with pytest.raises(VerifyError):
        validate_trial_inputs(ok_scores, 5, 2, 0, [1, 1])  # 重复
    with pytest.raises(VerifyError):
        validate_trial_inputs(ok_scores, 5, 2, True, [0])  # 布尔冒充整数


def test_validate_candidate_rules():
    n = 30
    triggers = {10, 20}
    protected = {20}
    used = {10}
    # 合法
    validate_candidate(10, n, triggers, protected, set(), template_count=0)
    # 帧号 < 7：凑不齐前 7 帧
    with pytest.raises(VerifyError):
        validate_candidate(6, n, {6}, set(), set(), 0)
    # 真口令帧
    with pytest.raises(VerifyError):
        validate_candidate(20, n, triggers, protected, set(), 0)
    # 已作过模板
    with pytest.raises(VerifyError):
        validate_candidate(10, n, triggers, set(), used, 1)
    # 不是当前触发帧
    with pytest.raises(VerifyError):
        validate_candidate(11, n, triggers, set(), set(), 0)
    # 模板达上限
    with pytest.raises(VerifyError):
        validate_candidate(10, n, triggers, set(), set(), MAX_TEMPLATES)
    # 越界
    with pytest.raises(VerifyError):
        validate_candidate(30, n, triggers, set(), set(), 0)


def test_max_size_performance():
    # 20 万帧 + 8 条模板：单遍 O(N) 回放，距离只在达阈未冷却帧上计算
    rng = random.Random(7)
    n = MAX_FRAMES
    scores = [rng.randint(0, 100) for _ in range(n)]
    tpls = [scores[i - 7 : i + 1] for i in (100, 1000, 5000, 20000, 40000, 80000, 120000, 160000)]
    t0 = time.perf_counter()
    res = replay(scores, 50, MAX_COOLDOWN, 30, [n - 1], tpls)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0
    assert res.distance_evaluations <= res.compared_frames * MAX_TEMPLATES
