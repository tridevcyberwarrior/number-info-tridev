"""
Search API — HF DuckDB Search
Develop by Tridev Cyber Warrior
Serverless-ready (Vercel / GitHub deploy)
"""
import asyncio
import json
import os
import time
from typing import Optional

import duckdb
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────
HF_INDEX_BASE = os.environ.get(
    "ICMR_HF_INDEX_BASE",
    "https://huggingface.co/buckets/godhosting/bakchodipromax",
).rstrip("/")

# Serverless friendly: keep threads low
THREADS = int(os.environ.get("ICMR_THREADS", "2"))
DUPLICATE_CAP = 2
TIMEOUT_DEFAULT = int(os.environ.get("ICMR_TIMEOUT", "25"))

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

REMOTE_INDEXES = {
    "phone": [f"{HF_INDEX_BASE}/idx_phone.{i}.parquet" for i in range(7)],
    "aadhar": [f"{HF_INDEX_BASE}/idx_aadhar.{i}.parquet" for i in range(7)],
}

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Search API — Tridev Cyber Warrior",
    description="HF-backed DuckDB search API. Develop by Tridev Cyber Warrior.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Connection (lazy, per-process) ────────────────────────────────────────
_conn: Optional[duckdb.DuckDBPyConnection] = None
_load_status = {
    "phone_loaded": False,
    "aadhar_loaded": False,
    "error": None,
    "init_time": None,
}


def _init_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is not None:
        return _conn

    con = duckdb.connect(database=":memory:")
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("SET threads = ?", [THREADS])
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("INSTALL httpfs; LOAD httpfs;")

    for kind, urls in REMOTE_INDEXES.items():
        view = f"people_{kind}"
        try:
            lst = ", ".join(f"'{u}'" for u in urls)
            con.execute(
                f"CREATE OR REPLACE VIEW {view} AS "
                f"SELECT * FROM read_parquet([{lst}], union_by_name=true)"
            )
            _load_status[f"{kind}_loaded"] = True
        except Exception as e:  # pragma: no cover
            _load_status["error"] = f"{kind}: {e}"

    _load_status["init_time"] = time.time()
    _conn = con
    return _conn


def _ready(kind: str) -> bool:
    return bool(_load_status.get(f"{kind}_loaded"))


# ── Helpers ───────────────────────────────────────────────────────────────
def _person_key(row: dict) -> tuple:
    ph = (row.get("phoneNumber") or "").strip()
    ad = (row.get("aadharNumber") or "").strip()
    if ph or ad:
        return (ph, ad)
    return (row.get("name") or "").strip(), (row.get("fathersName") or "").strip()


def _connected_numbers(row: dict) -> list[dict]:
    out, seen = [], set()
    for f in NUMBER_FIELDS:
        v = row.get(f)
        if not v:
            continue
        s = str(v).strip()
        if s and s not in seen:
            seen.add(s)
            out.append({"field": f, "value": s})
    return out


def _cap(rows: list[dict]) -> list[dict]:
    seen: dict[tuple, int] = {}
    out = []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            rec = dict(r)
            rec["connected_numbers"] = _connected_numbers(rec)
            out.append(rec)
    return out


def _field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")

    con = _init_conn()
    v = str(value).replace("'", "''")

    if field == "phoneNumber" and _ready("phone"):
        view = "people_phone"
    elif field == "aadharNumber" and _ready("aadhar"):
        view = "people_aadhar"
    else:
        return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}

    if mode == "exact":
        sql = f"SELECT * FROM {view} WHERE {field} = '{v}' LIMIT {limit * DUPLICATE_CAP + 20}"
    else:
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        sql = (
            f"SELECT * FROM {view} WHERE {field} ILIKE '%{v2}%' ESCAPE '\\' "
            f"LIMIT {limit * DUPLICATE_CAP + 20}"
        )

    try:
        rows = con.execute(sql).fetchall()
        cols = [d[0] for d in con.description]
        results = _cap([dict(zip(cols, r)) for r in rows])[:limit]
        return {"field": field, "value": value, "mode": mode,
                "count": len(results), "results": results}
    except Exception as e:
        return {"field": field, "value": value, "mode": mode,
                "count": 0, "results": [], "error": str(e)}


def _unified(q: str, limit: int) -> dict:
    q = q.strip()
    if not (q.isdigit() and len(q) >= 8):
        return {"query": q, "searched_fields": [], "count": 0, "results": []}

    rows, searched = [], []
    if _ready("phone"):
        r = _field_search("phoneNumber", q, "exact", limit)
        rows.extend(r["results"])
        searched.append("phoneNumber")
    if not rows and _ready("aadhar"):
        r = _field_search("aadharNumber", q, "exact", limit)
        rows.extend(r["results"])
        searched.append("aadharNumber")

    rows = _cap(rows)[:limit]
    return {"query": q, "searched_fields": searched,
            "count": len(rows), "results": rows}


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "app": "Search API",
        "author": "Develop by Tridev Cyber Warrior",
        "status": "running",
        "indexes": {"phone": _ready("phone"), "aadhar": _ready("aadhar")},
        "load_status": _load_status,
    }


@app.get("/health")
async def health():
    return {
        "status": "degraded" if _load_status["error"] else "healthy",
        "indexes": {"phone": _ready("phone"), "aadhar": _ready("aadhar")},
        "timestamp": time.time(),
    }


@app.get("/search")
async def search(
    q: Optional[str] = Query(None, description="Search query"),
    mobile: Optional[str] = Query(None, description="Mobile number"),
    field: Optional[str] = Query(None, description="Field name"),
    mode: str = Query("exact", pattern="^(exact|contains)$"),
    limit: int = Query(10, ge=1, le=200),
    timeout: int = Query(TIMEOUT_DEFAULT, ge=5, le=60),
    pretty: bool = Query(True),
):
    start = time.time()
    query = (q or mobile or "").strip()
    if not query:
        raise HTTPException(422, "Provide 'q' or 'mobile'")

    try:
        loop = asyncio.get_running_loop()
        if field:
            data = await asyncio.wait_for(
                loop.run_in_executor(None, _field_search, field, query, mode, limit),
                timeout=timeout,
            )
        else:
            data = await asyncio.wait_for(
                loop.run_in_executor(None, _unified, query, limit),
                timeout=timeout,
            )
    except asyncio.TimeoutError:
        return Response(
            content=json.dumps({"error": "timeout", "query": query}),
            status_code=504, media_type="application/json",
        )
    except Exception as e:
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500, media_type="application/json",
        )

    body = {
        "success": data.get("count", 0) > 0,
        "query": query,
        "count": data.get("count", 0),
        "results": data.get("results", []),
        "execution_time": round(time.time() - start, 3),
        "author": "Develop by Tridev Cyber Warrior",
    }
    return Response(
        content=json.dumps(body, indent=2 if pretty else None, ensure_ascii=False),
        media_type="application/json",
    )


@app.get("/fields")
async def fields():
    return {"fields": SEARCH_FIELDS, "number_fields": NUMBER_FIELDS}


if __name__ == "__main__":
    uvicorn.run("index:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", 8000)))
