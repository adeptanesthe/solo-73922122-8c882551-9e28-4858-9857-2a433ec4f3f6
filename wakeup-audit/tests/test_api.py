"""API 层测试：核对、CSV 下载、重新核对后旧链接失效（旧结论/旧下载撤下）。"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def post_check(receipts=None, **over):
    payload = {
        "scores": [5, 5, 0, 5, 5, 5],
        "threshold": 5,
        "cooldown": 1,
        "gap_start": 2,
        "gap_end": 4,
        # r=0 触发 [0,3,5]、r=1 触发 [1,3,5]；区间外回执 {0,5} 使 r=0 一致
        "receipts": receipts if receipts is not None else [0, 5],
    }
    payload.update(over)
    return client.post("/api/check", json=payload)


def test_health_and_limits():
    assert client.get("/api/health").json() == {"status": "ok"}
    limits = client.get("/api/limits").json()
    assert limits["max_frames"] == 200_000
    assert limits["max_cooldown"] == 2000


def test_check_ok_and_csv():
    resp = post_check()
    assert resp.status_code == 200
    data = resp.json()
    rid = data["result_id"]
    assert data["status"] == "ok"
    assert data["backfill"], "缺报区间内应有可补记帧"

    csv_resp = client.get(f"/api/results/{rid}/backfill.csv")
    assert csv_resp.status_code == 200
    text = csv_resp.content.decode("utf-8-sig")
    lines = text.strip().splitlines()
    assert lines[0] == "frame,score,status"
    assert len(lines) == 1 + len(data["backfill"])
    assert f'attachment; filename="backfill_{rid[:8]}.csv"' in csv_resp.headers["content-disposition"]


def test_conflict_has_no_csv_and_lists_rejections():
    # 回执帧 2 在区间内但永不达标（该帧分数为 0）
    resp = post_check(receipts=[2])
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "conflict"
    assert data["backfill"] == []
    assert len(data["rejections"]) == 2  # r = 0, 1
    csv_resp = client.get(f"/api/results/{data['result_id']}/backfill.csv")
    assert csv_resp.status_code == 409


def test_recheck_invalidates_old_result_and_download():
    first = post_check().json()
    old_id = first["result_id"]
    assert client.get(f"/api/results/{old_id}/backfill.csv").status_code == 200

    # 更正缺报区间后重新核对
    second = post_check(gap_start=1, gap_end=3).json()
    assert second["result_id"] != old_id

    # 旧结论数据与旧下载都必须撤下
    assert client.get(f"/api/results/{old_id}/backfill.csv").status_code == 410
    assert client.get(f"/api/results/{second['result_id']}/backfill.csv").status_code == 200


def test_bad_input_returns_400():
    resp = post_check(cooldown=9999)
    assert resp.status_code == 400
    assert "冷却帧数" in resp.json()["detail"]


def test_index_page_served():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "唤醒词缺报核对" in resp.text


# ---------------- 过滤试跑 ----------------
def trial_start_payload(**over):
    # 阈值 5、冷却 0；背景同形在帧 10、20，真口令异形在帧 30
    scores = [0] * 40
    shape = [5, 5, 5, 5, 5, 5, 5, 9]
    scores[3:11] = shape
    scores[13:21] = shape
    scores[23:31] = [9, 9, 9, 9, 9, 9, 9, 9]
    payload = {
        "scores": scores,
        "threshold": 5,
        "cooldown": 0,
        "tolerance": 0,
        "protected": [30],
    }
    payload.update(over)
    return payload


def test_trial_limits_endpoint():
    data = client.get("/api/limits").json()
    assert data["max_templates"] == 8
    assert data["template_window"] == 8


def test_trial_start_baseline_and_candidates():
    resp = client.post("/api/trial/start", json=trial_start_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["passed"] is True
    assert data["missed_protected"] == []
    assert data["template_count"] == 0
    # 原规则下背景帧 10、20 与真口令 30 都触发
    assert {10, 20, 30}.issubset(set(data["triggers"]))
    # 候选 = 未标注的当前触发帧（真口令 30 不可选）
    assert 10 in data["candidates"] and 20 in data["candidates"]
    assert 30 not in data["candidates"]
    assert data["stats"]["distance_evaluations"] == 0


def test_trial_template_accepted_blocks_repeated_background():
    start = client.post("/api/trial/start", json=trial_start_payload()).json()
    sid = start["session_id"]
    resp = client.post(f"/api/trial/{sid}/template", json={"frame": 10})
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] is True
    assert data["template_count"] == 1
    assert data["templates"][0]["source_frame"] == 10
    # 同形背景 10、20 被同一条模板拦下，真口令 30 保住
    assert {b["frame"] for b in data["blocked"]} == {10, 20}
    assert all(b["template_index"] == 0 for b in data["blocked"])
    assert 30 in data["triggers"]
    assert data["passed"] is True
    # 已作模板的帧 10 不再出现在候选里
    assert 10 not in data["candidates"]


def test_trial_template_rejected_keeps_previous_state():
    # 容差拉到极大：任何 8 帧窗口都命中模板，真口令 30 会被拦 -> 拒绝该模板
    start = client.post(
        "/api/trial/start", json=trial_start_payload(tolerance=10**9)
    ).json()
    sid = start["session_id"]
    before_triggers = start["triggers"]
    resp = client.post(f"/api/trial/{sid}/template", json={"frame": 10})
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] is False
    assert data["rejected_frame"] == 10
    assert data["candidate_missed_protected"] == [30]
    # 保留原有模板（0 条）和原结果
    assert data["template_count"] == 0
    assert data["triggers"] == before_triggers
    assert data["blocked"] == []
    assert data["passed"] is True
    assert data["missed_protected"] == []  # 当前生效结果仍然通过
    # 会话仍可继续使用：再加合法操作不受影响
    again = client.post(f"/api/trial/{sid}/template", json={"frame": 20})
    assert again.status_code == 200
    assert again.json()["accepted"] is False  # 大容差下同样漏真口令


def test_trial_candidate_validation_errors():
    start = client.post("/api/trial/start", json=trial_start_payload()).json()
    sid = start["session_id"]
    # 帧号 < 7：没有完整前 7 帧
    r = client.post(f"/api/trial/{sid}/template", json={"frame": 6})
    assert r.status_code == 400 and "前 7 帧" in r.json()["detail"]
    # 不是当前触发帧
    r = client.post(f"/api/trial/{sid}/template", json={"frame": 11})
    assert r.status_code == 400 and "不是当前回放的触发帧" in r.json()["detail"]
    # 真口令帧
    r = client.post(f"/api/trial/{sid}/template", json={"frame": 30})
    assert r.status_code == 400 and "真口令帧" in r.json()["detail"]
    # 先成功加一条，再重复同一帧
    ok = client.post(f"/api/trial/{sid}/template", json={"frame": 10})
    assert ok.status_code == 200 and ok.json()["accepted"] is True
    r = client.post(f"/api/trial/{sid}/template", json={"frame": 10})
    assert r.status_code == 400 and "已经作为模板" in r.json()["detail"]
    # 布尔帧号在模型层被拒
    r = client.post(f"/api/trial/{sid}/template", json={"frame": True})
    assert r.status_code == 422


def test_trial_template_capacity_eight():
    # 8 个背景触发帧：末帧 9 达标，各自窗口含独特标记 -> 窗口两两不同形，互不拦截；
    # 真口令帧 100 的窗口全为 9（异形），任何模板都不会误伤它。
    scores = [0] * 120
    for k, end in enumerate((10, 20, 30, 40, 50, 60, 70, 80)):
        scores[end] = 9
        scores[end - 1] = 1 + k  # 独特标记，保证 8 个窗口互不相同
    scores[93:101] = [9] * 8  # 真口令 100 的窗口（帧 93..100 全 9）
    payload = trial_start_payload(scores=scores, tolerance=0, protected=[100])
    start = client.post("/api/trial/start", json=payload).json()
    sid = start["session_id"]
    for end in (10, 20, 30, 40, 50, 60, 70, 80):
        r = client.post(f"/api/trial/{sid}/template", json={"frame": end})
        assert r.status_code == 200 and r.json()["accepted"] is True
    full = client.get(f"/api/trial/{sid}").json()
    assert full["template_count"] == 8
    assert 100 in full["triggers"]  # 真口令始终保住
    # 第 9 条：模板已达上限，优先报容量错误
    r = client.post(f"/api/trial/{sid}/template", json={"frame": 80})
    assert r.status_code == 400 and "最多" in r.json()["detail"]


def test_trial_start_validation():
    # 容差为负
    r = client.post("/api/trial/start", json=trial_start_payload(tolerance=-1))
    assert r.status_code == 400 and "容差" in r.json()["detail"]
    # 没有真口令帧
    r = client.post("/api/trial/start", json=trial_start_payload(protected=[]))
    assert r.status_code == 400 and "真口令" in r.json()["detail"]
    # 超帧数
    r = client.post(
        "/api/trial/start", json=trial_start_payload(scores=[1] * 200001)
    )
    assert r.status_code == 400 and "200000" in r.json()["detail"]
    # 原规则就漏掉真口令（真口令帧不达标）-> 不允许开始
    bad = trial_start_payload()
    bad["scores"][30] = 0
    r = client.post("/api/trial/start", json=bad)
    assert r.status_code == 400 and "漏掉真口令帧" in r.json()["detail"]
    # 布尔值不得冒充整数（pydantic 默认会把 true 转成 1，必须在模型层拦下）
    r = client.post("/api/trial/start", json=trial_start_payload(tolerance=True))
    assert r.status_code == 422
    r = client.post("/api/trial/start", json=trial_start_payload(protected=[True]))
    assert r.status_code == 422
    r = client.post(
        "/api/trial/start", json=trial_start_payload(threshold=True)
    )
    assert r.status_code == 422
    # 小数阈值由 pydantic 直接拒绝
    r = client.post("/api/trial/start", json=trial_start_payload(threshold=5.5))
    assert r.status_code == 422


def test_trial_session_eviction_returns_410():
    # 连续新建超过保留上限的会话后，最早会话失效（迟到结果作废）
    from app.main import MAX_SESSIONS

    first = client.post("/api/trial/start", json=trial_start_payload()).json()
    sid = first["session_id"]
    for _ in range(MAX_SESSIONS):
        client.post("/api/trial/start", json=trial_start_payload())
    r = client.post(f"/api/trial/{sid}/template", json={"frame": 10})
    assert r.status_code == 410
    r = client.get(f"/api/trial/{sid}")
    assert r.status_code == 410
