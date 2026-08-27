# --------------------------------------------------------------------------
# Repository copy of knowledge_injection/contract_index.py, unmodified apart
# from this header. Runs CPU-only against the archived corpus/spec files in
# this repository. See code/README.md for the module map.
# --------------------------------------------------------------------------
"""Derived API contract helpers.

The core API specification remains the standardized ``functions`` list.  This
module derives machine-readable contracts from function documentation and
object/table attributes.  It is called by ``library_knowledge.py`` so runtime
metadata and contracts are emitted as one derived artifact.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set


STOPWORDS = {
    "a",
    "ac",
    "all",
    "and",
    "at",
    "data",
    "for",
    "from",
    "in",
    "is",
    "net",
    "of",
    "or",
    "pu",
    "res",
    "result",
    "results",
    "table",
    "the",
    "to",
    "with",
}

QUERY_SYNONYMS = {
    "bus": ["bus", "buses"],
    "line": ["line", "lines"],
    "gen": ["generator", "generators", "generation"],
    "sgen": ["static generator", "static generators", "generation"],
    "trafo": ["transformer", "transformers", "trafo", "trafos"],
    "vm": ["voltage", "voltage magnitude"],
    "va": ["angle", "voltage angle"],
    "p": [],
    "pl": ["active power loss", "active power losses", "power loss", "power losses", "losses"],
    "q": [],
    "mw": [],
    "mvar": [],
    "loading": ["loading", "loaded", "overload"],
    "percent": ["percent", "%"],
    "ikss": ["short circuit current", "fault current"],
    "ka": ["current"],
    "cost": ["cost", "objective"],
    "scaling": ["scaling", "scale"],
    "in_service": ["in service", "out of service", "disconnect", "reconnect"],
}

GENERIC_QUERY_TERMS = {
    "a",
    "active power",
    "angle",
    "current",
    "degree",
    "ka",
    "mvar",
    "mw",
    "p",
    "percent",
    "pu",
    "q",
    "real power",
    "value",
    "va",
    "vm",
}


def _dedup(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        text = " ".join(str(item or "").split())
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _short(text: str, limit: int = 180) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _tokens(text: str) -> List[str]:
    pieces = re.split(r"[^A-Za-z0-9]+", str(text or "").lower())
    return [p for p in pieces if len(p) >= 2 and p not in STOPWORDS]


def _human_name(name: str) -> str:
    return " ".join(_tokens(name))


def _regex_phrase(phrase: str) -> str:
    words = _tokens(phrase)
    if not words:
        return ""
    return r"\b" + r"[_\s-]?".join(re.escape(w) for w in words) + r"\b"


def _near_pattern(left: str, right: str, distance: int = 80) -> str:
    lpat = _regex_phrase(left)
    rpat = _regex_phrase(right)
    if not lpat or not rpat:
        return ""
    return f"{lpat}.{{0,{distance}}}{rpat}"


def _table_phrases(table: str) -> List[str]:
    base = table[4:] if table.startswith("res_") else table
    pieces = [_human_name(base)]
    for token in _tokens(base):
        pieces.extend(QUERY_SYNONYMS.get(token, []))
    if "sc" in _tokens(base):
        pieces.extend(["short circuit", "fault"])
    return _dedup(pieces)


def _field_phrases(table: str, column: str, description: str) -> List[str]:
    column_tokens = _tokens(column)
    table_base = table[4:] if table.startswith("res_") else table
    table_tokens = set(_tokens(table_base))
    phrases: List[str] = []
    column_phrase = _human_name(column)
    if column_phrase and column_phrase not in GENERIC_QUERY_TERMS:
        phrases.append(column_phrase)

    lower_desc = str(description or "").lower()
    joined = "_".join(column_tokens)
    if column == "p_mw" or joined.endswith("_p_mw"):
        phrases.extend(["active power", "real power"])
        if table_tokens & {"gen", "sgen", "ext", "grid"}:
            phrases.extend(["power output", "output", "generation", "dispatch"])
        elif table_tokens & {"load", "motor", "storage", "ward", "xward"}:
            phrases.extend(["active power demand", "demand", "consumption"])
    if column == "q_mvar" or joined.endswith("_q_mvar"):
        phrases.extend(["reactive power"])
    if column == "pl_mw" or joined.endswith("_pl_mw"):
        phrases.extend(["active power loss", "active power losses", "power loss", "power losses", "losses"])
    if "loading" in column_tokens:
        phrases.extend(["loading", "loading percent", "loaded"])
    if column.startswith("vm_") or "voltage magnitude" in lower_desc:
        phrases.extend(["voltage", "voltage magnitude"])
    if column.startswith("va_") or "voltage angle" in lower_desc:
        phrases.extend(["voltage angle", "angle"])
    if "ikss" in column_tokens:
        phrases.extend(["ikss", "short circuit current", "fault current"])
    if column.startswith("i_") or column.endswith("_ka"):
        phrases.extend(["current"])
    if "cost" in lower_desc or column == "value":
        phrases.extend(["cost", "objective", "objective value"])

    for token in column_tokens:
        phrases.extend(QUERY_SYNONYMS.get(token, []))

    if re.search(r"\bactive\s+power\s+loss(?:es)?\b", lower_desc):
        phrases.extend(["active power losses", "power losses", "losses"])
    elif re.search(r"\bactive\s+power\s+flow\b", lower_desc):
        phrases.extend(["active power flow", "power flow"])
    elif re.search(r"\bactive\s+power\b", lower_desc):
        phrases.extend(["active power", "real power"])
    if re.search(r"\breactive\s+power\s+flow\b", lower_desc):
        phrases.extend(["reactive power flow"])
    elif re.search(r"\breactive\s+power\b", lower_desc):
        phrases.append("reactive power")
    if "short-circuit current" in lower_desc or "short circuit current" in lower_desc:
        phrases.extend(["short circuit current", "fault current"])
    if "voltage magnitude" in lower_desc:
        phrases.extend(["voltage", "voltage magnitude"])

    return _dedup(phrases)


def _informative_standalone_phrase(phrase: str) -> bool:
    norm = " ".join(_tokens(phrase))
    if not norm or norm in GENERIC_QUERY_TERMS:
        return False
    return norm in {"cost", "objective", "objective value"}


def _query_patterns_for_field(table: str, column: str, description: str) -> List[str]:
    table_phrases = _table_phrases(table)
    field_phrases = _field_phrases(table, column, description)
    candidates: List[str] = []
    for table_phrase in table_phrases:
        for field_phrase in field_phrases:
            candidates.extend([
                _near_pattern(table_phrase, field_phrase),
                _near_pattern(field_phrase, table_phrase),
            ])
    for field_phrase in field_phrases:
        if _informative_standalone_phrase(field_phrase):
            candidates.append(_regex_phrase(field_phrase))
    return _dedup(p for p in candidates if p)


def _code_patterns_for_field(access_pattern: str, column: str) -> List[str]:
    escaped_access = re.escape(access_pattern)
    escaped_col = re.escape(column)
    table_name = access_pattern.split(".")[-1]
    return [
        rf"{escaped_access}[^\n]{{0,120}}{escaped_col}",
        rf"{re.escape(table_name)}[^\n]{{0,120}}{escaped_col}",
    ]


def _unique_functions(functions: Sequence[Dict]) -> List[Dict]:
    by_name: Dict[str, Dict] = {}
    for item in functions or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if name and name not in by_name:
            by_name[name] = item
    return list(by_name.values())


def _attribute_contract(attribute: Dict) -> Dict:
    name = str(attribute.get("name") or "")
    access = str(attribute.get("access_pattern") or name)
    attr_type = str(attribute.get("type") or "attribute")
    data_structure = str(attribute.get("data_structure") or "")
    is_dataframe = "dataframe" in data_structure.lower()
    fields = [
        {
            "name": str(p.get("name") or ""),
            "type": str(p.get("type") or ""),
            "unit": str(p.get("unit") or ""),
            "description": _short(p.get("description") or "", 160),
        }
        for p in list(attribute.get("parameters") or [])
        if isinstance(p, dict) and p.get("name")
    ]
    field_names = [f["name"] for f in fields]
    lines = [
        f"{access} is a {data_structure or attr_type} ({attr_type}).",
    ]
    if field_names:
        shown = ", ".join(f"`{f}`" for f in field_names[:16])
        suffix = " ..." if len(field_names) > 16 else ""
        lines.append(f"Available fields include {shown}{suffix}.")
    if attr_type in {"input_table", "result_table"} and is_dataframe:
        lines.append(
            f"Use `{access}.loc[index, 'field']` or `{access}['field']` for DataFrame access."
        )
    elif attr_type in {"scalar_property", "result_table"} and not is_dataframe:
        lines.append(f"Read the scalar/object value directly from `{access}`.")
    return {
        "id": f"schema:{name}",
        "kind": "object_schema",
        "name": name,
        "access_pattern": access,
        "attribute_type": attr_type,
        "data_structure": data_structure,
        "description": _short(attribute.get("description") or "", 220),
        "fields": fields,
        "contract": lines,
        "source": "attribute_schema",
        "confidence": 0.9,
    }


def _field_contracts(attribute: Dict) -> List[Dict]:
    contracts: List[Dict] = []
    name = str(attribute.get("name") or "")
    access = str(attribute.get("access_pattern") or name)
    attr_type = str(attribute.get("type") or "")
    data_structure = str(attribute.get("data_structure") or "")
    is_dataframe = "dataframe" in data_structure.lower()
    is_result = attr_type == "result_table" or name.startswith("res_")
    is_input = attr_type == "input_table"
    for param in list(attribute.get("parameters") or []):
        if not isinstance(param, dict) or not param.get("name"):
            continue
        column = str(param.get("name"))
        desc = _short(param.get("description") or "", 180)
        desc_lower = desc.lower()
        if is_result and (
            ("constraint" in desc_lower or "limit" in desc_lower)
            and re.search(r"^(?:min|max)_", column)
        ):
            # Keep these fields in the table-level schema, but do not expose
            # them as output observables.  In many scientific APIs, limit or
            # constraint fields live alongside result fields but are not the
            # requested simulation output.
            continue
        kind = "output_observable" if is_result else "state_mutation" if is_input else "object_field"
        verb = "Read" if is_result else "Update" if is_input else "Access"
        read_expression = f"{access}['{column}']" if is_dataframe else access
        access_hint = (
            f"{verb} `{read_expression}` ({param.get('type') or 'value'})."
            if is_dataframe
            else f"{verb} `{access}` directly ({param.get('type') or data_structure or 'value'})."
        )
        contract = [
            access_hint,
        ]
        if desc:
            contract.append(desc)
        contracts.append({
            "id": f"{kind}:{name}.{column}",
            "kind": kind,
            "name": f"{name}.{column}",
            "table": name,
            "field": column,
            "access_pattern": access,
            "read_expression": read_expression,
            "query_patterns": _query_patterns_for_field(name, column, desc),
            "code_patterns": (
                [re.escape(access)]
                if not is_dataframe
                else _code_patterns_for_field(access, column)
            ),
            "contract": contract,
            "source": "attribute_schema",
            "confidence": 0.82 if kind == "output_observable" else 0.78,
        })
    return contracts


def _function_contract(func: Dict) -> Dict:
    name = str(func.get("name") or "")
    full_path = str(func.get("full_path") or name)
    params = [
        str(p.get("name"))
        for p in list(func.get("parameters") or [])
        if isinstance(p, dict) and p.get("name") and not str(p.get("name")).startswith("*")
    ]
    desc = _short(func.get("description") or "", 220)
    contract = []
    if desc:
        contract.append(desc)
    call_args = ", ".join(params[:6])
    if len(params) > 6:
        call_args += ", ..."
    contract.append(f"Call pattern: `{full_path}({call_args})`.")
    return {
        "id": f"function:{name}",
        "kind": "function_contract",
        "name": name,
        "full_path": full_path,
        "query_patterns": [_regex_phrase(name), _regex_phrase(desc)],
        "contract": [line for line in contract if line],
        "source": "function_doc",
        "confidence": 0.86,
    }


def derive_contract_index(
    docs: Dict,
    *,
    attribute_docs: Optional[Dict] = None,
    source_docs_path: str = "",
    attribute_source_path: str = "",
) -> Dict:
    """Return a derived contract-index dictionary."""
    functions = _unique_functions(docs.get("functions") or [])
    attributes = list((docs.get("attributes") or []))
    if attribute_docs:
        attributes = list(attribute_docs.get("attributes") or attributes)

    contracts: List[Dict] = []
    contracts.extend(_function_contract(func) for func in functions if func.get("name"))
    for attr in attributes:
        if not isinstance(attr, dict) or not attr.get("name"):
            continue
        contracts.append(_attribute_contract(attr))
        contracts.extend(_field_contracts(attr))

    by_kind: Dict[str, int] = {}
    for contract in contracts:
        by_kind[contract["kind"]] = by_kind.get(contract["kind"], 0) + 1

    return {
        "version": "contract_index_v1",
        "library": docs.get("library") or docs.get("library_name") or "unknown",
        "source_docs_path": source_docs_path,
        "attribute_source_path": attribute_source_path,
        "generated_from": {
            "functions": len(functions),
            "attributes": len(attributes),
        },
        "summary": {
            "total_contracts": len(contracts),
            "by_kind": by_kind,
        },
        "contracts": contracts,
    }
