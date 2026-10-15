"""Brace-matching JSON object extractor (string-aware)."""
import json


def json_objects(text):
    """All top-level {...} objects that json-parse, in order (brace matching, string-aware)."""
    out, depth, start, ins, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if ins:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                ins = False
            continue
        if ch == '"':
            ins = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    out.append(json.loads(text[start:i + 1]))
                except json.JSONDecodeError:
                    pass
    return out


def last_obj(text, key):
    objs = [o for o in json_objects(text) if key in o]
    return objs[-1] if objs else None
