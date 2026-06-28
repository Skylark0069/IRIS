import json
from typing import Any, Dict, Optional, Union, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


JSON_PATH = "skincare_compounds_info.json"
HTML_PATH = "IRIS_dynamic_fashion.html"


def load_json(json_path: str) -> Dict[str, Any]:
    """
    {
        "ID": {
            "SMILES": "...",
            "cid": "...",
            "CAS": "...",
            "name": "...",
            ...
        }
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


class CompoundSearchEngine:
    VALID_FIELDS = {"ID", "SMILES", "CID", "CAS", "NAME"}

    FIELD_TO_JSON_KEY = {
        "SMILES": "SMILES",
        "CID": "cid",
        "CAS": "CAS",
        "NAME": "name",
    }

    def __init__(self, data: Dict[str, Any], case_sensitive: bool = False):
        self.data = data
        self.case_sensitive = case_sensitive
        # {
        #     "ID": {"100": ["100"]},
        #     "CAS": {"50-00-0": ["123"]},
        #     "NAME": {"aspirin": ["456", "789"]},
        #     ...
        # }
        self.index: Dict[str, Dict[str, List[str]]] = {
            field: {} for field in self.VALID_FIELDS
        }

        self._build_index()

    def normalize(self, value: Any) -> Optional[str]:
        if value is None:
            return None

        value = str(value).strip()

        if not self.case_sensitive:
            value = value.lower()

        return value

    def _add_to_index(self, field: str, value: Any, compound_id: str) -> None:
        value_norm = self.normalize(value)

        if not value_norm:
            return

        if value_norm not in self.index[field]:
            self.index[field][value_norm] = []

        if compound_id not in self.index[field][value_norm]:
            self.index[field][value_norm].append(compound_id)

    def _build_index(self) -> None:
        for compound_id, entry in self.data.items():
            self._add_to_index("ID", compound_id, compound_id)

            for field, json_key in self.FIELD_TO_JSON_KEY.items():
                value = entry.get(json_key)

                if isinstance(value, list):
                    for item in value:
                        self._add_to_index(field, item, compound_id)
                else:
                    self._add_to_index(field, value, compound_id)

    def search(
        self,
        query: Union[str, int],
        query_type: Optional[str] = None
    ) -> Dict[str, Any]:
        query_norm = self.normalize(query)

        if not query_norm:
            raise ValueError("query can not be empty")

        if query_type is not None:
            query_type = query_type.upper()

            if query_type not in self.VALID_FIELDS:
                raise ValueError(
                    f"query_type must be: {sorted(self.VALID_FIELDS)}"
                )

            matched_ids = self.index[query_type].get(query_norm, [])

        else:
            matched_ids_set = set()

            for field in self.VALID_FIELDS:
                ids = self.index[field].get(query_norm, [])
                matched_ids_set.update(ids)

            matched_ids = list(matched_ids_set)

        results = {
            compound_id: self.data[compound_id]
            for compound_id in matched_ids
        }

        return {
            "ok": True,
            "query": query,
            "query_type": query_type or "AUTO",
            "count": len(results),
            "results": results,
        }

    def get_by_id(self, compound_id: Union[str, int]) -> Dict[str, Any]:
        result = self.search(compound_id, query_type="ID")
        return result

    def list_by_criteria(self, list_type: str) -> Dict[str, Any]:
        list_type = list_type.lower()
        valid_types = {"hpat", "tp_hpat", "cn_key", "eu_key", "jp_key", "kr_key", "us_key"}
        if list_type not in valid_types:
            raise ValueError(
                f"Unknown list_type: {list_type}. Must be one of {sorted(valid_types)}"
            )

        def _is_true(val: Any) -> bool:
            return str(val).strip().lower() == "true"

        results = {}
        for compound_id, entry in self.data.items():
            if list_type == "hpat":
                if _is_true(entry.get("HPAT")):
                    results[compound_id] = entry
            elif list_type == "tp_hpat":
                if _is_true(entry.get("TP-HPAT")):
                    results[compound_id] = entry
            elif list_type.endswith("_key"):
                area_code = list_type[:2].upper()  # cn → CN, eu → EU
                future_areas = entry.get("future_key_AREA") or []
                if area_code in future_areas:
                    results[compound_id] = entry

        compounds = []
        for compound_id, entry in results.items():
            compounds.append({
                "_id": compound_id,
                "SMILES": entry.get("SMILES", ""),
                "HPAT": entry.get("HPAT", "Unknown"),
                "TP-HPAT": entry.get("TP-HPAT", "Unknown"),
            })

        return {
            "ok": True,
            "type": list_type,
            "count": len(compounds),
            "compounds": compounds,
        }


    def search_patents(self, keyword: str) -> Dict[str, Any]:
        keyword_lower = (keyword or "").strip().lower()
        if not keyword_lower:
            raise ValueError("keyword can be empty")

        results: Dict[str, Dict[str, Any]] = {}

        for compound_id, entry in self.data.items():
            patents = entry.get("patents") or {}
            if not isinstance(patents, dict):
                continue

            matched = False
            for pat_num, pat_data in patents.items():
                if not isinstance(pat_data, dict):
                    continue

                title = str(pat_data.get("patent_title", "")).lower()
                if keyword_lower in title:
                    matched = True
                    break

                abstract = str(pat_data.get("patent_abstract", "")).lower()
                if keyword_lower in abstract:
                    matched = True
                    break

                matched_fns = pat_data.get("matched_functions") or []
                for fn in matched_fns:
                    fn_name = str(fn.get("function_name", "")).lower()
                    fn_evidence = str(fn.get("evidence", "")).lower()
                    if keyword_lower in fn_name or keyword_lower in fn_evidence:
                        matched = True
                        break
                if matched:
                    break

            if matched:
                results[compound_id] = {
                    "_id": compound_id,
                    "SMILES": entry.get("SMILES", ""),
                    "HPAT": entry.get("HPAT", "Unknown"),
                    "TP-HPAT": entry.get("TP-HPAT", "Unknown"),
                }

        compounds = list(results.values())
        # Sort: HPAT=True first
        compounds.sort(key=lambda c: str(c.get("HPAT", "")).strip().lower() != "true")

        return {
            "ok": True,
            "keyword": keyword,
            "count": len(compounds),
            "compounds": compounds,
        }

    @staticmethod
    def _normalize_function_name(raw: str) -> str:
        import re
        name = raw.strip()
        name = re.sub(r'^\d+[\.\-\s]*\s*', '', name)
        name = name.replace('_', ' ').replace('-', ' ')
        name = name.title()
        name = name.strip()
        return name

    def list_all_functions(self) -> Dict[str, Any]:
        # key = lowercase normalized, value = display name (title case)
        fn_map: Dict[str, str] = {}
        for compound_id, entry in self.data.items():
            patents = entry.get("patents") or {}
            if not isinstance(patents, dict):
                continue
            for pat_num, pat_data in patents.items():
                if not isinstance(pat_data, dict):
                    continue
                matched_fns = pat_data.get("matched_functions") or []
                for fn in matched_fns:
                    raw = str(fn.get("function_name", "")).strip()
                    if not raw:
                        continue
                    normalized = self._normalize_function_name(raw)
                    key = normalized.lower()
                    if key not in fn_map:
                        fn_map[key] = normalized

        fn_list = sorted(fn_map.values(), key=str.lower)
        return {
            "ok": True,
            "count": len(fn_list),
            "functions": fn_list,
        }

    def search_by_function(self, function_name: str) -> Dict[str, Any]:
        fn_lower = self._normalize_function_name(function_name).lower()
        if not fn_lower:
            raise ValueError("function_name can not be empty")

        results: Dict[str, Dict[str, Any]] = {}

        for compound_id, entry in self.data.items():
            patents = entry.get("patents") or {}
            if not isinstance(patents, dict):
                continue

            matched = False
            for pat_num, pat_data in patents.items():
                if not isinstance(pat_data, dict):
                    continue
                matched_fns = pat_data.get("matched_functions") or []
                for fn in matched_fns:
                    existing_fn = self._normalize_function_name(
                        str(fn.get("function_name", ""))
                    ).lower()
                    if existing_fn == fn_lower:
                        matched = True
                        break
                if matched:
                    break

            if matched:
                results[compound_id] = {
                    "_id": compound_id,
                    "SMILES": entry.get("SMILES", ""),
                    "HPAT": entry.get("HPAT", "Unknown"),
                    "TP-HPAT": entry.get("TP-HPAT", "Unknown"),
                }

        compounds = list(results.values())
        compounds.sort(key=lambda c: str(c.get("HPAT", "")).strip().lower() != "true")

        return {
            "ok": True,
            "function_name": self._normalize_function_name(function_name),
            "count": len(compounds),
            "compounds": compounds,
        }


class SearchRequest(BaseModel):
    query: Union[str, int]
    query_type: Optional[str] = None


app = FastAPI(
    title="Skincare Compounds Search API",
    description="Local backend API for compound search",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


data = load_json(JSON_PATH)
search_engine = CompoundSearchEngine(data)


@app.get("/")
def root():
    return FileResponse(HTML_PATH, media_type="text/html")


@app.get("/api/info")
def api_info():
    return {
        "message": "Skincare Compounds Search API is running",
        "available_endpoints": [
            "/api/compound?q=100&field=ID",
            "/api/compound/100",
            "/api/search",
            "/api/compounds/hpat",
            "/api/compounds/tp_hpat",
            "/api/compounds/cn_key",
            "/api/compounds/eu_key",
            "/api/compounds/jp_key",
            "/api/compounds/kr_key",
            "/api/compounds/us_key",
            "/api/patents/search?q=moisturizing",
            "/api/functions",
            "/api/functions/moisturizing",
        ],
    }


@app.get("/api/compound")
def query_compound_api(
    q: Union[str, int] = Query(..., description="查询值，例如 ID、SMILES、CID、CAS、NAME"),
    field: Optional[str] = Query(
        None,
        description="查询字段，可选：ID / SMILES / CID / CAS / NAME。不传则自动搜索所有字段。"
    )
):
    """
    GET 查询接口。

    示例：
        /api/compound?q=100&field=ID
        /api/compound?q=50-00-0&field=CAS
        /api/compound?q=aspirin&field=NAME
        /api/compound?q=aspirin
    """
    try:
        return search_engine.search(query=q, query_type=field)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/search")
def query_compound_post_api(request: SearchRequest):
    """
    POST 查询接口，适合前端用 JSON 请求。

    请求体示例：
    {
        "query": "100",
        "query_type": "ID"
    }
    """
    try:
        return search_engine.search(
            query=request.query,
            query_type=request.query_type
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/compound/{compound_id}")
def get_compound_by_id(compound_id: str):
    """
    根据 ID 直接查询。

    示例：
        /api/compound/100
    """
    try:
        result = search_engine.get_by_id(compound_id)

        if result["count"] == 0:
            raise HTTPException(status_code=404, detail="未找到对应 ID")

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/compounds/{list_type}")
def list_compounds(list_type: str):
    """
    按条件列出化合物。

    示例：
        /api/compounds/hpat      — HPAT 为 True
        /api/compounds/tp_hpat   — TP-HPAT 为 True
        /api/compounds/cn_key    — future_key_AREA 包含 CN
        /api/compounds/eu_key    — future_key_AREA 包含 EU
        /api/compounds/jp_key    — future_key_AREA 包含 JP
        /api/compounds/kr_key    — future_key_AREA 包含 KR
        /api/compounds/us_key    — future_key_AREA 包含 US
    """
    try:
        return search_engine.list_by_criteria(list_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/patents/search")
def search_patents_api(
    q: str = Query(..., description="专利搜索关键词，模糊匹配 patent_title / patent_abstract / function_name / evidence"),
):
    """
    模糊搜索专利。

    在所有化合物的所有专利中搜索关键词。
    匹配范围：patent_title, patent_abstract, matched_functions(function_name, evidence)

    示例：
        /api/patents/search?q=moisturizing
        /api/patents/search?q=anti-aging
    """
    try:
        return search_engine.search_patents(q)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/functions")
def list_functions_api():
    """
    列出所有专利中的唯一 function_name。

    示例：
        /api/functions
    """
    try:
        return search_engine.list_all_functions()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/functions/{function_name}")
def search_by_function_api(function_name: str):
    """
    根据 function_name 查找匹配的化合物。

    在所有专利的 matched_functions 中精确匹配 function_name。

    示例：
        /api/functions/moisturizing
        /api/functions/anti-aging
    """
    try:
        return search_engine.search_by_function(function_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))