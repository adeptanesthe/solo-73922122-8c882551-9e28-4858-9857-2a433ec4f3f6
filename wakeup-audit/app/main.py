"""FastAPI 入口：核对接口、补记 CSV 下载、静态页面。

每次重新核对都会使上一次的结论和下载链接失效（旧链接返回 410），
前端也会在重新核对时清空旧结论与下载入口。
"""

from __future__ import annotations

import csv
import io
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .engine import MAX_COOLDOWN, MAX_FRAMES, VerifyError, verify
from .filter_trial import (
    MAX_TEMPLATES,
    TEMPLATE_WINDOW,
    ReplayResult,
    replay,
    validate_candidate,
    validate_trial_inputs,
)

app = FastAPI(title="唤醒缺报核对", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"


class CheckRequest(BaseModel):
    scores: list[int] = Field(..., description="逐帧模型整数分数，帧号从 0 起")
    threshold: int
    cooldown: int
    gap_start: int
    gap_end: int
    receipts: list[int] = Field(default_factory=list, description="已收到的触发帧号")

    @field_validator("threshold", "cooldown", "gap_start", "gap_end", mode="before")
    @classmethod
    def reject_bool_scalar(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("必须是整数，不能是布尔值。")
        return v

    @field_validator("scores", "receipts", mode="before")
    @classmethod
    def reject_bool_list(cls, v: object) -> object:
        if isinstance(v, list) and any(isinstance(x, bool) for x in v):
            raise ValueError("帧号与分数必须是整数，不能包含布尔值。")
        return v


class TrialStartRequest(BaseModel):
    scores: list[int] = Field(..., description="逐帧模型整数分数，最多 200000 帧")
    threshold: int
    cooldown: int
    tolerance: int = Field(..., description="非负整数：8 帧逐项绝对差之和的上限")
    protected: list[int] = Field(..., description="至少一个要保住的真口令帧")

    @field_validator("threshold", "cooldown", "tolerance", mode="before")
    @classmethod
    def reject_bool_scalar(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("必须是整数，不能是布尔值。")
        return v

    @field_validator("scores", "protected", mode="before")
    @classmethod
    def reject_bool_list(cls, v: object) -> object:
        if isinstance(v, list) and any(isinstance(x, bool) for x in v):
            raise ValueError("帧号与分数必须是整数，不能包含布尔值。")
        return v


class TrialTemplateRequest(BaseModel):
    frame: int = Field(..., description="当前触发帧，且须有完整的前 7 帧")

    @field_validator("frame", mode="before")
    @classmethod
    def reject_bool_frame(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("模板帧号必须是整数，不能是布尔值。")
        return v


class ResultStore:
    """只保留最近一次核对结果；重新核对即作废旧结论与旧下载。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_id: str | None = None
        self._results: dict[str, dict] = {}

    def put(self, result: dict) -> str:
        with self._lock:
            self._results.clear()  # 作废旧结论与旧下载链接
            rid = uuid.uuid4().hex
            self._results[rid] = result
            self._current_id = rid
            return rid

    def get(self, rid: str) -> dict | None:
        with self._lock:
            return self._results.get(rid)


store = ResultStore()

# ---------------- 过滤试跑会话 ----------------
# 每次"开始试跑"生成一个会话；会话只存活在内存，输入改动即由前端结束。
# 只保留最近 MAX_SESSIONS 个会话，超出按最早会话淘汰（迟到结果自然作废）。
MAX_SESSIONS = 16


class TrialSession:
    def __init__(
        self,
        scores: list[int],
        threshold: int,
        cooldown: int,
        tolerance: int,
        protected: list[int],
    ) -> None:
        self.lock = threading.Lock()
        self.scores = scores
        self.threshold = threshold
        self.cooldown = cooldown
        self.tolerance = tolerance
        self.protected = sorted(protected)
        self.protected_set = set(protected)
        self.templates: list[list[int]] = []
        self.used_frames: set[int] = set()
        self.used_frames_ordered: list[int] = []
        self.last_result: ReplayResult | None = None
        self.created_at = time.time()


class TrialStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, TrialSession] = OrderedDict()

    def put(self, sid: str, session: TrialSession) -> None:
        with self._lock:
            self._sessions[sid] = session
            self._sessions.move_to_end(sid)
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)

    def get(self, sid: str) -> TrialSession | None:
        with self._lock:
            session = self._sessions.get(sid)
            if session is not None:
                self._sessions.move_to_end(sid)
            return session


trial_store = TrialStore()


def replay_payload(session: TrialSession, result: ReplayResult) -> dict:
    """把回放结果与会话状态拼成前端需要的 JSON。"""
    return {
        "threshold": session.threshold,
        "cooldown": session.cooldown,
        "tolerance": session.tolerance,
        "protected": session.protected,
        "frame_count": len(session.scores),
        "templates": [
            {
                "index": idx,
                "source_frame": session.used_frames_ordered[idx],
                "window": tpl,
            }
            for idx, tpl in enumerate(session.templates)
        ],
        "template_count": len(session.templates),
        "template_limit": MAX_TEMPLATES,
        "template_window": TEMPLATE_WINDOW,
        "passed": result.passed,
        "missed_protected": result.missed_protected,
        "triggers": result.triggers,
        "trigger_scores": {str(f): session.scores[f] for f in result.triggers},
        "blocked": result.blocked,
        "stats": {
            "trigger_count": len(result.triggers),
            "blocked_count": len(result.blocked),
            "protected_count": len(session.protected),
            "missed_count": len(result.missed_protected),
            "distance_evaluations": result.distance_evaluations,
            "compared_frames": result.compared_frames,
        },
        # 工程师下一轮可选的未标注触发帧（当前回放触发、非真口令帧、未作过模板且帧号 >= 7）
        "candidates": [
            f
            for f in result.triggers
            if f >= TEMPLATE_WINDOW - 1
            and f not in session.protected_set
            and f not in session.used_frames
        ],
    }


@app.post("/api/trial/start")
def trial_start(req: TrialStartRequest) -> dict:
    """开始一次过滤试跑：校验输入并按原规则（无模板）从可触发状态回放。"""
    try:
        n = validate_trial_inputs(
            req.scores, req.threshold, req.cooldown, req.tolerance, req.protected
        )
        result = replay(
            scores=req.scores,
            threshold=req.threshold,
            cooldown=req.cooldown,
            tolerance=req.tolerance,
            protected=req.protected,
            templates=[],
        )
        # 无模板可拦截，漏掉真口令帧说明原规则本身就保不住，试跑无法成立
        if not result.passed:
            raise VerifyError(
                "按原规则回放已漏掉真口令帧 "
                + ", ".join(str(f) for f in result.missed_protected)
                + "，无法开始试跑，请检查阈值、冷却或真口令帧设置。"
            )
    except VerifyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session = TrialSession(
        scores=req.scores,
        threshold=req.threshold,
        cooldown=req.cooldown,
        tolerance=req.tolerance,
        protected=req.protected,
    )
    session.last_result = result
    sid = uuid.uuid4().hex
    trial_store.put(sid, session)
    return {"session_id": sid, **replay_payload(session, result)}


@app.post("/api/trial/{sid}/template")
def trial_add_template(sid: str, req: TrialTemplateRequest) -> dict:
    """在试跑会话中加入一条模板并从原分数重放；不通过则保留原模板与原结果。"""
    session = trial_store.get(sid)
    if session is None:
        raise HTTPException(status_code=410, detail="该试跑已结束或已被新一轮试跑替换，请重新开始试跑。")

    with session.lock:
        if session.last_result is None:  # 理论上不会发生，start 时必有结果
            raise HTTPException(status_code=409, detail="试跑状态异常，请重新开始试跑。")
        n = len(session.scores)
        try:
            validate_candidate(
                frame=req.frame,
                n=n,
                current_triggers=set(session.last_result.triggers),
                protected_set=session.protected_set,
                used_frames=session.used_frames,
                template_count=len(session.templates),
            )
        except VerifyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        window = session.scores[req.frame - TEMPLATE_WINDOW + 1 : req.frame + 1]
        candidate_templates = session.templates + [list(window)]
        result = replay(
            scores=session.scores,
            threshold=session.threshold,
            cooldown=session.cooldown,
            tolerance=session.tolerance,
            protected=session.protected,
            templates=candidate_templates,
        )

        if not result.passed:
            # 漏掉真口令帧：拒绝该模板，保留原有模板和结果
            return {
                "session_id": sid,
                "accepted": False,
                "rejected_frame": req.frame,
                "candidate_missed_protected": result.missed_protected,
                **replay_payload(session, session.last_result),
            }

        # 通过才更新模板
        session.templates.append(list(window))
        session.used_frames.add(req.frame)
        session.used_frames_ordered.append(req.frame)
        session.last_result = result
        return {
            "session_id": sid,
            "accepted": True,
            **replay_payload(session, result),
        }


@app.get("/api/trial/{sid}")
def trial_status(sid: str) -> dict:
    """查询试跑会话当前状态（刷新页面后的兜底；输入改动后前端应直接结束试跑）。"""
    session = trial_store.get(sid)
    if session is None:
        raise HTTPException(status_code=410, detail="该试跑已结束或已被新一轮试跑替换。")
    with session.lock:
        assert session.last_result is not None
        return {"session_id": sid, **replay_payload(session, session.last_result)}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/limits")
def limits() -> dict:
    return {
        "max_frames": MAX_FRAMES,
        "max_cooldown": MAX_COOLDOWN,
        "max_templates": MAX_TEMPLATES,
        "template_window": TEMPLATE_WINDOW,
    }


@app.post("/api/check")
def check(req: CheckRequest) -> dict:
    try:
        result = verify(
            scores=req.scores,
            threshold=req.threshold,
            cooldown=req.cooldown,
            gap_start=req.gap_start,
            gap_end=req.gap_end,
            receipts=req.receipts,
        )
    except VerifyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = result.__dict__
    rid = store.put(payload)
    return {"result_id": rid, **payload}


@app.get("/api/results/{rid}/backfill.csv")
def download_backfill(rid: str) -> StreamingResponse:
    result = store.get(rid)
    if result is None:
        # 链接曾经存在但已被新一轮核对撤下，或根本不存在
        raise HTTPException(status_code=410, detail="该补记下载已失效：请重新核对。")
    if result["status"] != "ok":
        raise HTTPException(status_code=409, detail="冲突状态下没有可补记列表。")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["frame", "score", "status"])
    for row in result["backfill"]:
        writer.writerow([row["frame"], row["score"], row["status"]])
    data = buf.getvalue().encode("utf-8-sig")

    filename = f"backfill_{rid[:8]}.csv"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
