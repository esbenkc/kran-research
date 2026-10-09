"""Build the blinded proof corpus (AI vs human) for the proof-elegance project.

    python3 projects/proof-elegance/build_corpus.py fetch   # cache raw sources into data/raw/ (idempotent)
    python3 projects/proof-elegance/build_corpus.py          # rebuild data/corpus.jsonl + data/{ai,human}/<id>.txt

Selection (which papers, why, and per-paper extraction hints) is hardcoded in data/selection.json.
`build` reads only data/selection.json and data/raw/; it never touches the network.

Extraction rule (same code path for both sides):
  * Flatten the LaTeX (\\input/\\include), strip comments, expand short user macros.
  * Main theorem = selection `main_label`, else the first Theorem-type environment.
  * Main proof = the proof attached to it (proof env right after it, `\\begin{proof}[Proof of \\ref{..}]`,
    or a section titled "Proof of ..."), overridable with `proof_label` / `proof_section`.
  * Proof region R = main statement + every main-matter section after the introduction up to the
    section holding the main proof (later sections and appendices only if the proof refers into them),
    plus intro subsections titled overview/strategy/outline. Discussion/acknowledgement/reference
    sections are dropped.
  * If R renders to <= CAP characters it is output whole. Otherwise it is truncated to CAP by priority:
    statement, main proof, overview, section skeleton, statements of results the main proof uses
    (transitively, nearest first), then their proofs; omitted spans become "[...]".
  * Blinding removes names, affiliations, acknowledgements, dates, arXiv ids, URLs, citation keys
    (citations become [n]) and any sentence mentioning AI tools / Lean / formalization.
    The mathematics and prose are not rewritten.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import collections
import json
import re
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
RAW = DATA / "raw"
CAP = 25_000
OAI_REPO = Path("/tmp/oai-math")  # only used by `fetch`
UA = {"User-Agent": "kran-research proof-elegance corpus builder (mailto:esben@kran.ai)"}

# --------------------------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------------------------


def fetch(sel: dict) -> None:
    (RAW / "ai").mkdir(parents=True, exist_ok=True)
    (RAW / "human").mkdir(parents=True, exist_ok=True)
    for it in sel["ai"]:
        dst = RAW / "ai" / it["slug"]
        if dst.exists():
            continue
        src = OAI_REPO / "preprints" / it["slug"]
        for f in src.rglob("*"):
            if f.is_file() and f.suffix in {".tex", ".bib", ".bbl", ".sty", ".cls", ".md"}:
                out = dst / f.relative_to(src)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out)
        print("cached ai", it["slug"])
    for it in sel["human"]:
        vid = it["arxiv_version"]
        p = RAW / "human" / f"{vid}.eprint"
        if not p.exists() or p.stat().st_size == 0:
            req = urllib.request.Request(f"https://arxiv.org/e-print/{vid}", headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                p.write_bytes(r.read())
            print("downloaded", vid)
            time.sleep(3.5)  # arXiv politeness
        unpack_eprint(p, RAW / "human" / vid)


def unpack_eprint(p: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.mkdir(parents=True)
    data = p.read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            for m in tf.getmembers():
                if m.isfile() and not m.name.startswith("/") and ".." not in m.name:
                    tf.extract(m, dst)
            return
    except tarfile.ReadError:
        pass
    (dst / "main.tex").write_bytes(data)


# --------------------------------------------------------------------------------------------
# LaTeX utilities
# --------------------------------------------------------------------------------------------


def read_text(p: Path) -> str:
    b = p.read_bytes()
    for enc in ("utf-8", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "ignore")


def strip_comments(s: str) -> str:
    out = []
    for line in s.split("\n"):
        i = 0
        while True:
            j = line.find("%", i)
            if j < 0:
                out.append(line)
                break
            k, bs = j - 1, 0
            while k >= 0 and line[k] == "\\":
                bs += 1
                k -= 1
            if bs % 2 == 0:
                out.append(line[:j])
                break
            i = j + 1
    s = "\n".join(out)
    s = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", "", s, flags=re.S)
    s = re.sub(r"\\iffalse\b.*?\\fi\b", "", s, flags=re.S)
    return s


def find_main(root: Path) -> Path:
    cands = []
    for f in root.rglob("*.tex"):
        t = read_text(f)
        if re.search(r"^[^%\n]*\\begin\{document\}", t, re.M):
            cands.append(f)
    if not cands:
        raise RuntimeError(f"no main tex in {root}")
    pref = [f for f in cands if f.stem.lower() in {"main", "paper", "ms", "article", "manuscript"}]
    pool = pref or cands
    return max(pool, key=lambda f: f.stat().st_size)


INPUT_RE = re.compile(r"\\(input|include|subfile)\s*(?:\{([^}]*)\}|\s+([^\s\\{}]+))")


def flatten(path: Path, root: Path, depth: int = 0) -> str:
    s = strip_comments(read_text(path))
    if depth > 6:
        return s

    def rep(m: re.Match) -> str:
        name = (m.group(2) or m.group(3) or "").strip()
        for base in (path.parent, root):
            for cand in (base / name, base / (name + ".tex")):
                if cand.is_file() and cand.suffix in {".tex", ".ltx", ""} or (cand.is_file() and cand.suffix not in {".bib", ".sty", ".cls", ".bbl", ".pdf", ".png"}):
                    if cand.suffix in {".bib", ".sty", ".cls", ".pdf", ".png", ".jpg"}:
                        continue
                    return flatten(cand, root, depth + 1)
        return ""

    return INPUT_RE.sub(rep, s)


def brace_arg(s: str, i: int) -> tuple[str, int] | None:
    """s[i] (after skipping spaces) must be '{'; return (content, index after closing brace)."""
    n = len(s)
    while i < n and s[i] in " \t\n":
        i += 1
    if i >= n or s[i] != "{":
        return None
    depth = 0
    j = i
    while j < n:
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
        j += 1
    return None


def opt_arg(s: str, i: int) -> tuple[str, int] | None:
    n = len(s)
    k = i
    while k < n and s[k] in " \t":
        k += 1
    if k >= n or s[k] != "[":
        return None
    depth = 0
    j = k
    while j < n:
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
            if depth == 0:
                return s[k + 1 : j], j + 1
        j += 1
    return None


def remove_cmd(s: str, names: list[str], keep_content: bool = False, nargs: int = 1) -> str:
    pat = re.compile(r"\\(" + "|".join(re.escape(n) for n in names) + r")\*?(?![A-Za-z@])")
    out, i = [], 0
    while True:
        m = pat.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i : m.start()])
        j = m.end()
        o = opt_arg(s, j)
        if o:
            j = o[1]
        contents = []
        for _ in range(nargs):
            a = brace_arg(s, j)
            if not a:
                break
            contents.append(a[0])
            j = a[1]
        if keep_content and contents:
            out.append(contents[-1])
        i = j
    return "".join(out)


# --------------------------------------------------------------------------------------------
# macros and theorem environments
# --------------------------------------------------------------------------------------------

DEFAULT_THM = {
    "theorem": "Theorem", "thm": "Theorem", "maintheorem": "Theorem", "mainthm": "Theorem", "theo": "Theorem",
    "lemma": "Lemma", "lem": "Lemma", "lm": "Lemma", "sublemma": "Lemma", "proposition": "Proposition",
    "prop": "Proposition", "pro": "Proposition", "corollary": "Corollary", "cor": "Corollary", "coro": "Corollary",
    "claim": "Claim", "fact": "Fact", "observation": "Observation", "definition": "Definition", "defn": "Definition",
    "dfn": "Definition", "defi": "Definition", "conjecture": "Conjecture", "conj": "Conjecture", "remark": "Remark",
    "rem": "Remark", "rmk": "Remark", "example": "Example", "ex": "Example", "question": "Question", "problem": "Problem",
    "assumption": "Assumption", "hypothesis": "Hypothesis", "notation": "Notation", "construction": "Construction",
}
PROOF_ENVS = {"proof", "pf", "proof*", "Proof", "proofof", "prf", "pfof"}
RESULT_KINDS = re.compile(r"theorem|lemma|proposition|corollary|claim|fact|observation|sublemma|satz|lemme|théorème", re.I)
SKIP_EXPAND = re.compile(r"\\(?:newcommand|renewcommand|def|let|ifx|else|fi|csname|expandafter|makeatletter|ifmmode|edef|gdef|xdef|futurelet|relax|DeclareMathOperator)(?![A-Za-z])|@")


def parse_defs(src: str) -> tuple[dict, dict, set]:
    macros: dict[str, tuple[int, str | None, str]] = {}
    thm: dict[str, str] = dict(DEFAULT_THM)
    proofs = set(PROOF_ENVS)
    for m in re.finditer(r"\\(?:re)?newcommand\*?|\\providecommand\*?|\\DeclareRobustCommand\*?", src):
        j = m.end()
        a = brace_arg(src, j)
        if a:
            name, j = a[0].strip(), a[1]
        else:
            mm = re.match(r"\s*(\\[A-Za-z@]+)", src[j:])
            if not mm:
                continue
            name, j = mm.group(1), j + mm.end()
        n, default = 0, None
        o = opt_arg(src, j)
        if o:
            try:
                n = int(o[0].strip())
            except ValueError:
                continue
            j = o[1]
            o2 = opt_arg(src, j)
            if o2:
                default, j = o2[0], o2[1]
        b = brace_arg(src, j)
        if not b or not re.fullmatch(r"\\[A-Za-z]+", name):
            continue
        macros[name[1:]] = (n, default, b[0])
    for m in re.finditer(r"\\def\s*\\([A-Za-z]+)((?:#\d)*)\s*\{", src):
        b = brace_arg(src, m.end() - 1)
        if b:
            macros[m.group(1)] = (len(m.group(2)) // 2, None, b[0])
    for m in re.finditer(r"\\DeclareMathOperator(\*?)\s*\{?\s*\\([A-Za-z]+)\s*\}?", src):
        b = brace_arg(src, m.end())
        if b:
            macros[m.group(2)] = (0, None, "\\operatorname%s{%s}" % (m.group(1), b[0]))
    for m in re.finditer(r"\\(?:new|sp)theorem\*?\s*\{([^}]+)\}\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", src):
        thm[m.group(1).strip()] = re.sub(r"\\[A-Za-z]+\s*|[{}]", "", m.group(2)).strip() or m.group(1)
    for m in re.finditer(r"\\declaretheorem\s*(?:\[([^\]]*)\])?\s*\{([^}]+)\}", src):
        nm = re.search(r"name\s*=\s*\{?([^,}\]]+)", m.group(1) or "")
        thm[m.group(2).strip()] = (nm.group(1).strip() if nm else m.group(2).strip().capitalize())
    for m in re.finditer(r"\\newenvironment\s*\{([^}]+)\}", src):
        name = m.group(1).strip()
        tail = src[m.end() : m.end() + 200]
        if re.search(r"\\begin\{proof\}|\\textit\{Proof|\\emph\{Proof|\{\\it Proof|\\noindent\s*\{?\\bf Proof", tail):
            proofs.add(name)
    # theorem-like envs defined via \newenvironment wrapping a theorem env
    for m in re.finditer(r"\\newenvironment\s*\{([^}]+)\}\s*(?:\[[^\]]*\])*\s*\{\s*\\begin\{([A-Za-z*]+)\}", src):
        if m.group(2) in thm and m.group(1) not in thm:
            thm[m.group(1)] = thm[m.group(2)]
    # unicode or math-heavy macros are fine; drop ones we must not expand
    clean = {k: v for k, v in macros.items() if len(v[2]) <= 160 and not SKIP_EXPAND.search(v[2]) and k not in {"proof", "qed", "label", "ref", "cite"}}
    return clean, thm, proofs


def expand_macros(s: str, macros: dict) -> str:
    if not macros:
        return s
    names = sorted(macros, key=len, reverse=True)
    pat = re.compile(r"\\(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z@])")
    for _ in range(4):
        changed = False
        out, i = [], 0
        while True:
            m = pat.search(s, i)
            if not m:
                out.append(s[i:])
                break
            out.append(s[i : m.start()])
            n, default, body = macros[m.group(1)]
            j = m.end()
            args = []
            if default is not None:
                o = opt_arg(s, j)
                if o:
                    args.append(o[0])
                    j = o[1]
                else:
                    args.append(default)
            ok = True
            while len(args) < n:
                a = brace_arg(s, j)
                if a:
                    args.append(a[0])
                    j = a[1]
                else:
                    mm = re.match(r"\s*(\\[A-Za-z]+|[^\s{}\\])", s[j:])
                    if not mm:
                        ok = False
                        break
                    args.append(mm.group(1))
                    j += mm.end()
            if not ok:
                out.append(m.group(0))
                i = m.end()
                continue
            rep = body
            for k, a in enumerate(args, 1):
                rep = rep.replace("#%d" % k, a)
            # keep a separator if the macro was followed by a letter (\R x -> \mathbb{R} x)
            if j < len(s) and s[j].isalpha() and rep and rep[-1].isalpha() and re.search(r"\\[A-Za-z]+$", rep):
                rep += " "
            out.append(rep)
            i = j
            changed = True
        s = "".join(out)
        if not changed:
            break
    return s


# --------------------------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------------------------

ENV_TOK = re.compile(r"\\(begin|end)\s*\{([^}]+)\}")
SEC_RE = re.compile(r"\\(section|subsection|subsubsection)\*?\s*(?=[\[{])")
REF_RE = re.compile(r"\\(?:[cC]ref|[cC]pageref|ref|eqref|autoref|Autoref|nameref|labelcref|thmref|Ref|vref|Vref|cref\*|Cref\*)\s*\{([^}]*)\}")
LABEL_RE = re.compile(r"\\label\s*\{([^}]*)\}")
CITE_RE = re.compile(r"\\(?:cite|citet|citep|citealt|citeauthor|citeyear|parencite|textcite|autocite|footcite|nocite|Cite|citen|upcite|cites)\*?\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}")
DISPLAY_ENVS = re.compile(r"\\begin\{(equation|align|gather|multline|eqnarray|flalign|alignat|displaymath|dmath)\*?\}")
DROP_SEC = re.compile(r"acknowledg|thanks|reference|bibliograph|conclu|open (problem|question)|further (work|question|direction|remark)|future|final remark|concluding|discussion|questions|outlook|funding|data availability|declaration|competing", re.I)
OVERVIEW_SEC = re.compile(r"overview|strategy|outline|sketch|idea|heuristic|roadmap|road map|organi[sz]ation of the proof|main ingredients|structure of the proof|proof method|our approach|method", re.I)
INTRO_SEC = re.compile(r"introduction|intro|background|main result|statement|^results$|overview of results", re.I)
APP_SEC = re.compile(r"^\W*(some\s+|further\s+|other\s+)?(applications?|consequences?|corollaries|generali[sz]ations?|extensions?\b|variants?|possible variants|related (results|problems)|remarks|examples?\b|counterexamples to|beyond)", re.I)


def own_labels(body: str) -> list[str]:
    """Labels that belong to an environment itself, not to nested displays/environments inside it."""
    masked = body
    for _ in range(3):
        masked2 = re.sub(r"\\begin\{([^}]+)\}.*?\\end\{\1\}", lambda m: " " * len(m.group(0)), masked, flags=re.S)
        masked2 = re.sub(r"\\\[.*?\\\]", lambda m: " " * len(m.group(0)), masked2, flags=re.S)
        if masked2 == masked:
            break
        masked = masked2
    return [m.group(1).strip() for m in LABEL_RE.finditer(masked)]


class Doc:
    def __init__(self, body: str, thm: dict, proofs: set):
        self.s = body
        self.thm = thm
        self.proofs = proofs
        self.envs: list[dict] = []
        self.sections: list[dict] = []
        self._parse()

    def _parse(self) -> None:
        s = self.s
        stack = []
        for m in ENV_TOK.finditer(s):
            kind, name = m.group(1), m.group(2).strip()
            if name not in self.thm and name not in self.proofs:
                continue
            if kind == "begin":
                stack.append((name, m.start(), m.end()))
            else:
                for k in range(len(stack) - 1, -1, -1):
                    if stack[k][0] == name:
                        nm, st, bend = stack.pop(k)
                        o = opt_arg(s, bend)
                        body_start = o[1] if o else bend
                        body = s[body_start : m.start()]
                        labs = own_labels(body)
                        self.envs.append(dict(
                            env=nm, name=self.thm.get(nm, "Proof"), is_proof=nm in self.proofs, opt=o[0] if o else "",
                            start=st, end=m.end(), body_start=body_start, body=body,
                            label=labs[0] if labs else None, labels=labs, depth=k,
                        ))
                        break
        self.envs.sort(key=lambda e: e["start"])
        # nesting depth = number of enclosing envs
        for e in self.envs:
            e["depth"] = sum(1 for f in self.envs if f["start"] < e["start"] and f["end"] >= e["end"] and f is not e)
        appendix_at = s.find("\\appendix")
        for m in SEC_RE.finditer(s):
            j = m.end()
            o = opt_arg(s, j)
            if o:
                j = o[1]
            a = brace_arg(s, j)
            if not a:
                continue
            title = a[0]
            lab = LABEL_RE.match(s[a[1] : a[1] + 200].lstrip()) if s[a[1] : a[1] + 200].lstrip().startswith("\\label") else None
            self.sections.append(dict(level={"section": 1, "subsection": 2, "subsubsection": 3}[m.group(1)], title=title,
                                      start=m.start(), head_end=a[1], label=lab.group(1) if lab else None,
                                      appendix=appendix_at >= 0 and m.start() > appendix_at))
        for i, sec in enumerate(self.sections):
            nxt = [t["start"] for t in self.sections[i + 1 :] if t["level"] <= sec["level"]]
            sec["end"] = nxt[0] if nxt else len(s)
        # numbering
        self.labels: dict[str, tuple[str, str]] = {}
        n = 0
        for e in self.envs:
            if e["is_proof"]:
                continue
            n += 1
            e["num"] = str(n)
            for lab in e["labels"]:
                self.labels[lab] = (e["name"], e["num"])
        cnt = [0, 0, 0]
        for sec in self.sections:
            if sec["level"] == 1:
                cnt = [cnt[0] + 1, 0, 0]
            elif sec["level"] == 2:
                cnt[1] += 1
                cnt[2] = 0
            else:
                cnt[2] += 1
            sec["num"] = ".".join(str(c) for c in cnt[: sec["level"]])
            if sec["label"]:
                self.labels[sec["label"]] = ("Section", sec["num"])
        q = 0
        for m in LABEL_RE.finditer(s):
            lab = m.group(1).strip()
            if lab in self.labels:
                continue
            q += 1
            self.labels[lab] = ("eq", str(q))
            # remember where the equation label sits
        self.label_pos = {m.group(1).strip(): m.start() for m in LABEL_RE.finditer(s)}

    # ---- helpers
    def statements(self):
        return [e for e in self.envs if not e["is_proof"]]

    def env_of_label(self, lab: str):
        for e in self.envs:
            if lab in e["labels"] and not e["is_proof"]:
                return e
        return None

    def section_at(self, pos: int, level: int = 1):
        best = None
        for sec in self.sections:
            if sec["level"] <= level and sec["start"] <= pos < sec["end"]:
                best = sec if (best is None or sec["level"] >= best["level"]) else best
        return best

    def proof_of(self, st: dict):
        """Proof env attached to statement `st`."""
        for lab in st.get("labels", []):
            for e in self.envs:
                if e["is_proof"] and e["opt"]:
                    head = re.split(r"\b(?:from|assuming|using|given|via|modulo|based on|conditional on)\b", e["opt"], flags=re.I)[0]
                    if lab in refs_in(head):
                        return e
        for e in self.envs:
            if e["is_proof"] and e["start"] >= st["end"] and not e["opt"].strip():
                gap = self.s[st["end"] : e["start"]]
                if len(re.sub(r"\s|\\label\{[^}]*\}|\\(medskip|smallskip|bigskip|noindent|vspace\{[^}]*\})", "", gap)) < 300 and e["depth"] == st["depth"]:
                    return e
                return None
            if e["is_proof"] and e["start"] >= st["end"]:
                gap = self.s[st["end"] : e["start"]]
                if len(re.sub(r"\s", "", gap)) < 300 and not RESULT_KINDS.search(e["opt"] or "") and "\\ref" not in e["opt"]:
                    return e
                return None
            if not e["is_proof"] and e["start"] >= st["end"] and e["depth"] <= st["depth"]:
                return None
        return None


# --------------------------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------------------------


def is_theorem_kind(e: dict) -> bool:
    return bool(re.search(r"theorem|main|satz|théorème|thm", e["name"] + " " + e["env"], re.I)) and not re.search(r"conj|question|problem", e["name"], re.I)


def locate(doc: Doc, it: dict) -> dict:
    ex = it.get("extract", {})
    sts = doc.statements()
    main = None
    if ex.get("main_label"):
        main = doc.env_of_label(ex["main_label"])
        if main is None:
            raise RuntimeError(f"main_label {ex['main_label']} not found")
    elif ex.get("main_index") is not None:
        main = sts[ex["main_index"]]
    else:
        generic = [e for e in sts if re.fullmatch(r"(main\s+)?(theorem|thm|satz|théorème)(\s+[A-Z0-9]+)?\*?", e["name"].strip(), re.I)
                   and not re.search(r"\\cite|\[\d", e["opt"])]
        labelled = [e for e in generic if any(re.search(r"main", lab, re.I) for lab in e["labels"])]
        cands = labelled or generic or [e for e in sts if is_theorem_kind(e)]
        main = cands[0] if cands else (sts[0] if sts else None)
    proof = None
    proof_section = None
    method = None
    if ex.get("proof_section"):
        rx = re.compile(ex["proof_section"], re.I)
        secs = [sec for sec in doc.sections if rx.search(sec["title"])]
        if not secs:
            raise RuntimeError("proof_section not found: " + ex["proof_section"])
        proof_section, method = secs[0], "section(override)"
    elif ex.get("proof_label"):
        st = doc.env_of_label(ex["proof_label"])
        proof = doc.proof_of(st) if st else None
        if proof is None:
            raise RuntimeError("proof_label has no proof: " + ex["proof_label"])
        method = "proof_label(override)"
    elif ex.get("proof_regex"):
        rx = re.compile(ex["proof_regex"], re.I | re.S)
        pr = [e for e in doc.envs if e["is_proof"] and rx.search(e["opt"] + " " + e["body"][:200])]
        if not pr:
            raise RuntimeError("proof_regex no match")
        proof, method = pr[0], "proof_regex(override)"
    elif main is not None:
        proof = doc.proof_of(main)
        method = "attached" if proof else None
        if proof is None and main.get("label"):
            # a later restatement referencing the main label, or a proof titled with the main label
            for e in doc.envs:
                if e["start"] > main["end"] and not e["is_proof"] and re.search(r"\{\s*" + re.escape(main["label"]) + r"\s*\}", e["opt"]):
                    proof = doc.proof_of(e)
                    if proof:
                        method = "restatement"
                        break
        if proof is None:
            rx = re.compile(r"proof of (the )?(main|theorem|thm)", re.I)
            for e in doc.envs:
                if e["is_proof"] and e["start"] > main["end"] and rx.search(e["opt"]) and (not main.get("label") or main["label"] in e["opt"] or "main" in e["opt"].lower() or not re.search(r"\\ref|\\cref", e["opt"], re.I)):
                    proof, method = e, "titled-proof"
                    break
        if proof is None:
            rx = re.compile(r"proof of (the )?(main|theorem|thm|\\ref|\\cref)|the proof of (the )?main|proofs? of (the )?main", re.I)
            for sec in doc.sections:
                if sec["start"] > main["start"] and rx.search(sec["title"]):
                    proof_section, method = sec, "section"
                    break
        if proof is None and proof_section is None and main.get("labels"):
            # AI-style "... This proves Theorem~\ref{thm:main}." at the end of the section that proves it
            first = [x for x in doc.sections if x["level"] == 1]
            intro_end = first[0]["end"] if first and first[0]["start"] <= main["start"] < first[0]["end"] else main["end"]
            labs = "|".join(re.escape(lab) for lab in main["labels"])
            hits = [m.start() for m in re.finditer(r"\\(?:c|C|auto)?ref\s*\{(?:" + labs + r")\}", doc.s)
                    if m.start() >= intro_end and re.search(r"prov|proof|complet|conclud|finish|establish|deduc|follows|implies|yield", doc.s[max(0, m.start() - 120) : m.start()], re.I)]
            if hits:
                pos = hits[-1]
                cands = [x for x in doc.sections if x["start"] <= pos < x["end"]]
                if cands:
                    proof_section, method = max(cands, key=lambda x: x["level"]), "concluding-ref"
    if main is None:
        raise RuntimeError("no main theorem")
    return dict(main=main, proof=proof, proof_section=proof_section, method=method or "none")


def refs_in(text: str) -> list[str]:
    out = []
    for m in REF_RE.finditer(text):
        for lab in m.group(1).split(","):
            lab = lab.strip()
            if lab and lab not in out:
                out.append(lab)
    return out


def build_region(doc: Doc, loc: dict, it: dict) -> dict:
    """Return the proof region as an ordered list of spans (start, end, role, depth)."""
    s = doc.s
    main, proof, psec = loc["main"], loc["proof"], loc["proof_section"]
    secs1 = [x for x in doc.sections if x["level"] == 1]
    intro = None
    if secs1 and (INTRO_SEC.search(secs1[0]["title"]) or main["start"] < secs1[0]["end"]):
        intro = secs1[0]
    if it.get("extract", {}).get("no_intro"):
        intro = None
    ppos = (proof["start"] if proof else psec["start"] if psec else None)
    pend = (proof["end"] if proof else psec["end"] if psec else None)
    # dependency closure from the main proof: statements it cites, their proofs, and so on
    main_text = s[ppos:pend] if ppos is not None else ""
    depth: dict[int, int] = {}
    frontier = [(lab, 1) for lab in refs_in(main_text)]
    seen: set[str] = set()
    sec_proofs: list[tuple[int, dict]] = []  # (depth, section) proving a cited statement
    while frontier:
        lab, d = frontier.pop(0)
        if lab in seen:
            continue
        seen.add(lab)
        st = doc.env_of_label(lab)
        if st is None or st is main:
            continue
        depth.setdefault(id(st), d)
        pr = doc.proof_of(st)
        txt = st["body"] + (pr["body"] if pr else "")
        if pr is None:
            for sec in doc.sections:
                if any(re.search(r"\{\s*" + re.escape(lb) + r"\s*\}", sec["title"]) for lb in st["labels"]) and re.search(r"proof|deduc|reduc", sec["title"], re.I):
                    sec_proofs.append((d, sec))
                    txt += s[sec["start"] : sec["end"]]
        for l2 in refs_in(txt):
            frontier.append((l2, d + 1))
    dep_envs = [e for e in doc.statements() if id(e) in depth]
    # region: top-level sections after intro up to the one holding the main proof
    body_start = intro["end"] if intro else 0
    region_secs = []
    psec1 = doc.section_at(ppos, 1) if ppos is not None else None
    dep_secs = {id(doc.section_at(e["start"], 1)) for e in dep_envs}
    dep_secs |= {id(doc.section_at(sec["start"], 1)) for _, sec in sec_proofs}
    dep_secs |= {id(doc.section_at(doc.label_pos[lab], 1)) for lab in seen if lab in doc.label_pos}
    for sec in secs1:
        if sec is intro or DROP_SEC.search(sec["title"]):
            continue
        if sec["start"] < body_start:
            continue
        tail_kind = APP_SEC.search(sec["title"]) and not it.get("extract", {}).get("region_all")
        if (sec["appendix"] or tail_kind) and id(sec) not in dep_secs and sec is not psec1:
            continue
        if it.get("extract", {}).get("drop_sections") and re.search(it["extract"]["drop_sections"], sec["title"], re.I):
            continue
        region_secs.append(sec)
    spans = []
    spans.append((main["start"], main["end"], "main", 0))
    if intro is not None:
        for sub in doc.sections:
            if sub["level"] >= 2 and intro["start"] <= sub["start"] < intro["end"] and OVERVIEW_SEC.search(sub["title"]):
                spans.append((sub["start"], sub["end"], "overview", 0))
        if ppos is not None and intro["start"] <= ppos < intro["end"]:
            spans.append((ppos, pend, "mainproof", 0))
        for e in dep_envs:
            if intro["start"] <= e["start"] < intro["end"] and e is not main:
                pr = doc.proof_of(e)
                spans.append((e["start"], pr["end"] if pr else e["end"], "dep", depth[id(e)]))
    if not secs1:
        spans.append((max(main["end"], 0), len(s), "body", 0))
    if it.get("extract", {}).get("whole_body"):
        spans.append((0, len(s), "body", 0))
    for sec in region_secs:
        spans.append((sec["start"], sec["end"], "body", 0))
    spans = merge_spans(spans)
    return dict(spans=spans, deps=dep_envs, depth=depth, ppos=ppos, pend=pend, region_secs=region_secs, sec_proofs=sec_proofs)


def merge_spans(spans):
    spans = sorted(spans)
    out = []
    for a, b, role, d in spans:
        if out and a < out[-1][1]:
            pa, pb, prole, pd = out[-1]
            out[-1] = (pa, max(pb, b), prole if prole in ("main", "mainproof") else role, min(pd, d))
        else:
            out.append((a, b, role, d))
    return out


# --------------------------------------------------------------------------------------------
# rendering + blinding
# --------------------------------------------------------------------------------------------

DROP_CMDS0 = ["maketitle", "tableofcontents", "noindent", "medskip", "smallskip", "bigskip", "newpage", "clearpage", "pagebreak",
              "linebreak", "nopagebreak", "qedhere", "qed", "allowdisplaybreaks", "centering", "par", "indent", "hfill", "vfill",
              "phantomsection", "sloppy", "fussy", "newline", "break", "small", "normalsize", "footnotesize", "large", "Large",
              "scriptsize", "raggedright", "frenchspacing", "appendix", "endproof", "qedsymbol", "BlackBox", "hfil", "goodbreak",
              "displaybreak", "leavevmode", "mbox{}", "relax", "protect", "bfseries", "itshape", "normalfont", "selectfont"]
DROP_CMDS1 = ["label", "index", "vspace", "hspace", "thanks", "footnotemark", "setcounter", "addtocounter", "setlength", "addcontentsline",
              "bibliographystyle", "bibliography", "pagestyle", "thispagestyle", "markboth", "address", "email", "curraddr", "urladdr",
              "keywords", "subjclass", "date", "author", "title", "affil", "dedicatory", "includegraphics", "hypersetup", "renewcommand",
              "newcommand", "numberwithin", "graphicspath", "pdfbookmark", "definecolor", "color", "linenumbers", "AtEndDocument",
              "mathcode", "ack", "orcid", "titlerunning", "authorrunning", "institute", "shortauthors", "MSC", "PACS", "arxiv"]
TEXT_FMT = ["emph", "textit", "textbf", "textsl", "textsc", "texttt", "underline", "textup", "textnormal", "textmd", "uline", "textsf", "mbox", "hbox", "textrm"]
AI_TELL = re.compile(
    r"OpenAI|\bGPT\b|GPT-\d|ChatGPT|Codex|\bLLMs?\b|language models?|AI[- ]generated|machine[- ]generated|model[- ]generated|"
    r"generated by (an? )?(AI|model|system|machine)|artificial intelligence|\bAI (system|model|tool|assistant)s?\b|"
    r"\bLean\b|Lean ?4|[Mm]athlib|formally[- ]verified|machine[- ]checked|proof assistant|\bComparator\b|"
    r"this (manuscript|paper|article) was (written|produced|generated|prepared)|github\.com",
)
FORMAL_TELL = re.compile(r"formali[sz](ed|ation)", re.I)
FORMAL_CTX = re.compile(r"Lean|Coq|Isabelle|computer|machine|verif|assistant|software|code", re.I)
ACK_TELL = re.compile(
    r"\b(was|is|were|are|been) (partially |partly |also |generously |in part )?(supported|funded) by\b|\bgrant\b|\bthank(?!s to\b)|\bgrateful|"
    r"acknowledg|\bfellowship\b|\bhospitality\b|\bbegan investigating\b|\bthe (first|second|third|fourth|last) (named )?author\b",
    re.I,
)
PROTECT = re.compile(r"Hahn[-–—]+Banach|Green'?s (function|formula|theorem|identit\w*|kernel)|Green[-–—]+Tao")
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"

ACCENTS = {"\"": "\u0308", "'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303", "H": "\u030b", "v": "\u030c", "c": "\u0327", "u": "\u0306", "=": "\u0304", ".": "\u0307", "k": "\u0328"}


def accents(t: str) -> str:
    """LaTeX text accents -> unicode (so names and words read normally); math is untouched in practice."""
    import unicodedata

    def rep(m: re.Match) -> str:
        return unicodedata.normalize("NFC", (m.group(3) or m.group(4)) + ACCENTS[m.group(1) or m.group(2)])

    t = re.sub(r"\\(?:([\"'`^~=.])|([Hvcuk])(?![A-Za-z]))\s*(?:\{\\?([A-Za-z])\}|\\?([A-Za-z]))", rep, t)
    for k, v in {"\\o ": "ø", "\\o{}": "ø", "\\O ": "Ø", "\\ss ": "ß", "\\ss{}": "ß", "\\l ": "ł", "\\L ": "Ł", "\\aa ": "å", "{\\o}": "ø", "{\\l}": "ł", "{\\ss}": "ß", "{\\aa}": "å"}.items():
        t = t.replace(k, v)
    return t


class Renderer:
    def __init__(self, doc: Doc, authors: list[str]):
        self.doc = doc
        self.cites: dict[str, int] = {}
        self.authors = authors

    def cite(self, m: re.Match) -> str:
        nums = []
        for k in m.group(1).split(","):
            k = k.strip()
            if not k:
                continue
            if k not in self.cites:
                self.cites[k] = len(self.cites) + 1
            nums.append(str(self.cites[k]))
        opt = re.findall(r"\[([^\]]*)\]", m.group(0).split("{")[0])
        opt = [o for o in opt if o.strip()]
        return "[" + ", ".join(nums) + ("; " + opt[-1] if opt else "") + "]"

    def ref(self, m: re.Match) -> str:
        cmd = m.group(0)
        parts = []
        for lab in m.group(1).split(","):
            kind, num = self.doc.labels.get(lab.strip(), ("", "??"))
            if cmd.startswith(("\\cref", "\\Cref", "\\autoref", "\\Autoref", "\\labelcref", "\\vref", "\\Vref")) and kind:
                parts.append(("(%s)" % num) if kind == "eq" else f"{kind} {num}")
            elif cmd.startswith("\\eqref"):
                parts.append("(%s)" % num)
            else:
                parts.append(num)
        return ", ".join(parts)

    def render(self, a: int, b: int) -> str:
        doc = self.doc
        t = doc.s[a:b]
        # environment heads (theorem-like and proofs), keep math envs
        out, i = [], 0
        for m in ENV_TOK.finditer(t):
            name = m.group(2).strip()
            if name in doc.thm or name in doc.proofs:
                out.append(t[i : m.start()])
                if m.group(1) == "begin":
                    o = opt_arg(t, m.end())
                    optxt = o[0] if o else ""
                    if name in doc.proofs:
                        out.append("\n\nProof" + (f" ({optxt})" if optxt.strip() else "") + ". ")
                    else:
                        num = next((e.get("num", "") for e in doc.envs if e["start"] == a + m.start()), "")
                        out.append("\n\n" + f"{doc.thm[name]} {num}".strip() + (f" ({optxt})" if optxt.strip() else "") + ". ")
                    i = o[1] if o else m.end()
                else:
                    out.append(" \u220e\n\n" if name in doc.proofs else "\n\n")
                    i = m.end()
            elif name in {"itemize", "enumerate", "description", "center", "flushleft", "flushright", "quote", "quotation", "minipage", "small", "footnotesize"}:
                out.append(t[i : m.start()])
                i = m.end()
                if m.group(1) == "begin":
                    o = opt_arg(t, i)
                    if o:
                        i = o[1]
                    if name == "minipage":
                        bb = brace_arg(t, i)
                        if bb:
                            i = bb[1]
            elif name in {"abstract"} and m.group(1) == "begin":
                out.append(t[i : m.start()])
                end = t.find("\\end{abstract}", m.end())
                i = end + len("\\end{abstract}") if end >= 0 else m.end()
            else:
                continue
        out.append(t[i:])
        t = "".join(out)
        # sections
        res, i = [], 0
        for m in SEC_RE.finditer(t):
            j = m.end()
            o = opt_arg(t, j)
            if o:
                j = o[1]
            bb = brace_arg(t, j)
            if not bb:
                continue
            num = next((x["num"] for x in doc.sections if x["start"] == a + m.start()), "")
            hashes = "#" * (1 + {"section": 1, "subsection": 2, "subsubsection": 3}[m.group(1)])
            res.append(t[i : m.start()])
            res.append(f"\n\n{hashes} {num} {bb[0].strip()}\n\n")
            i = bb[1]
        res.append(t[i:])
        t = "".join(res)
        t = re.sub(r"\\paragraph\*?\s*\{([^{}]*)\}", r"\n\n\1 ", t)
        t = re.sub(r"\\subparagraph\*?\s*\{([^{}]*)\}", r"\n\n\1 ", t)
        # figures / tables / pictures
        t = re.sub(r"\\begin\{(figure|figure\*|wrapfigure|tikzpicture|picture|pspicture|SCfigure)\}.*?\\end\{\1\}", "\n[figure]\n", t, flags=re.S)
        t = re.sub(r"\\begin\{(thebibliography)\}.*?\\end\{\1\}", "", t, flags=re.S)
        t = re.sub(r"\\begin\{(acknowledgments|acknowledgements|acknowledgment|acknowledgement|ack|acks)\}.*?\\end\{\1\}", "", t, flags=re.S)
        t = re.sub(r"\\(begin|end)\{(table|table\*)\}(\[[^\]]*\])?", "", t)
        t = CITE_RE.sub(self.cite, t)
        t = REF_RE.sub(self.ref, t)
        # labelled displays keep their number so in-text "(7)" references stay resolvable
        t = re.sub(r"\\begin\{(equation|align|gather|multline|eqnarray|flalign|alignat)(\*?)\}.*?\\end\{\1\2\}", self.tag_labels, t, flags=re.S)
        t = remove_cmd(t, ["footnote"])
        t = remove_cmd(t, ["href"], keep_content=True, nargs=2)
        t = remove_cmd(t, ["url"], keep_content=False)
        t = remove_cmd(t, DROP_CMDS1)
        t = remove_cmd(t, ["caption"], keep_content=True)
        t = remove_cmd(t, TEXT_FMT, keep_content=True)
        t = re.sub(r"\{\\(?:em|it|bf|sl|sc|rm|tt|sf)\s+", "{", t)
        t = re.sub(r"\\(" + "|".join(DROP_CMDS0) + r")(?![A-Za-z@])\s?", "", t)
        t = re.sub(r"\\item\s*\[([^\]]*)\]", r"\n- \1 ", t)
        t = re.sub(r"\\item(?![A-Za-z])", "\n- ", t)
        t = accents(t)
        t = re.sub(r"\\(today)\b", "", t)
        # one delimiter convention on both sides (\( \) -> $ $, \[ \] -> $$ $$): house style is a source tell
        t = re.sub(r"(?<!\\)\\\(|(?<!\\)\\\)", "$", t)
        t = re.sub(r"(?<!\\)\\\[|(?<!\\)\\\]", "$$", t)
        t = t.replace("~", " ").replace("\\@", "").replace("\\ ", " ")
        return t

    def tag_labels(self, m: re.Match) -> str:
        def tag(mm: re.Match) -> str:
            kind, num = self.doc.labels.get(mm.group(1).strip(), ("", ""))
            return "\\tag{%s}" % num if kind == "eq" and num else ""

        return LABEL_RE.sub(tag, m.group(0))
        return t

    def blind(self, t: str) -> tuple[str, list[str]]:
        removed = []
        t = re.sub(r"arXiv[: ]*\d{4}\.\d{4,5}(v\d+)?|arXiv[: ]*[a-z\-]+/\d{7}(v\d+)?", "", t)
        t = re.sub(r"https?://\S+|www\.\S+", "", t)
        t = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "", t)
        t = re.sub(r"\b(" + MONTHS + r")\s+(\d{1,2},?\s+)?\d{4}\b", "", t)
        # explicit calendar years in prose ("in 2019", "(2019)", "2019 version") on both sides
        t = re.sub(r"\b(in|since|until|by|of|from|around|circa|early|late|mid|year) (?:19|20)\d{2}\b(?=\s*[,.;:)\]]|\s+(?:version|paper|preprint|article|work|survey|edition|note)s?\b)", r"\1 [year]", t)
        t = re.sub(r"([A-Za-z],? )\((?:19|20)\d{2}\)", r"\1", t)
        # acknowledgement paragraphs
        paras = re.split(r"(\n[ \t]*\n)", t)
        keep = []
        for p in paras:
            if not has_math(p) and (re.match(r"\W*(acknowledg|funding|we (would like to |also |warmly )*thank|the authors? (would like to |also )*thank)", p, re.I) or len(ACK_TELL.findall(p)) >= 2):
                removed.append(p.strip()[:300])
                continue
            keep.append(p)
        t = "".join(keep)
        # sentence/line-level removal of AI/tooling tells and acknowledgement sentences
        parts = re.split(r"((?<=[.!?])[ \t]+|\n)", t)
        keep = []
        for p in parts:
            if p and not re.fullmatch(r"\s+", p) and (
                AI_TELL.search(p)
                or (not is_structural(p) and ((FORMAL_TELL.search(p) and FORMAL_CTX.search(p)) or (ACK_TELL.search(p) and not has_math(p))))
                or re.match(r"\s*#+ [\d.]*\s*(acknowledg|funding)", p, re.I)
            ):
                removed.append(p.strip()[:300])
                continue
            keep.append(p)
        t = "".join(keep)
        # author names (human side): full names first, then surnames; eponyms in PROTECT are kept
        saved: list[str] = []

        def stash(m: re.Match) -> str:
            saved.append(m.group(0))
            return f"\x00{len(saved) - 1}\x00"

        t = PROTECT.sub(stash, t)
        for full in sorted(self.authors, key=len, reverse=True):
            toks = full.split()
            if not toks:
                continue
            sur = re.escape(toks[-1])
            first = re.escape(toks[0])
            t = re.sub(first + r"\.?\s+(?:[A-Z]\.\s*)*" + sur, "[author]", t)
            t = re.sub(r"(?<![A-Za-z])" + re.escape(toks[0][0]) + r"\.\s*(?:[A-Z]\.\s*)*" + sur + r"(?![A-Za-z])", "[author]", t)
            if len(toks[-1]) >= 3:
                t = re.sub(r"(?<![A-Za-z\\])" + sur + r"(?![a-z])", "[author]", t)
        t = re.sub(r"\x00(\d+)\x00", lambda m: saved[int(m.group(1))], t)
        if "[author]" in t:  # biographical / credit sentences about the authors carry no mathematics
            parts = re.split(r"((?<=[.!?])[ \t]+|\n)", t)
            keep = []
            for p in parts:
                if p and "[author]" in p and not has_math(p) and not is_structural(p):
                    removed.append(p.strip()[:300])
                    continue
                keep.append(p)
            t = "".join(keep)
        return t, removed


HEADER_RE = re.compile(r"^\s*(#+ |(Theorem|Lemma|Proposition|Corollary|Claim|Definition|Conjecture|Remark|Fact|Observation|Proof|Main Theorem)\b[^.]{0,120}\.)")


def has_math(p: str) -> bool:
    return "$" in p or "\\[" in p or "\\(" in p or "\\begin{" in p


def is_structural(p: str) -> bool:
    """Section headings and theorem headers are never dropped by the sentence filters."""
    return bool(HEADER_RE.match(p))


def normalize(t: str) -> str:
    t = t.replace("\r", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip() + "\n"


# --------------------------------------------------------------------------------------------
# per-paper pipeline
# --------------------------------------------------------------------------------------------


def load_doc(root: Path) -> tuple[Doc, str, str]:
    main = find_main(root)
    flat = flatten(main, main.parent)
    pre, _, rest = flat.partition("\\begin{document}")
    body = rest.split("\\end{document}")[0] if rest else flat
    macros, thm, proofs = parse_defs(pre + "\n" + body)
    body = expand_macros(body, macros)
    # drop front matter + bibliography from the body we analyse
    body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", body, flags=re.S)
    body = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", body, flags=re.S)
    body = re.sub(r"\\(bibliography|bibliographystyle|printbibliography)\b(\{[^}]*\})?", "", body)
    for c in ("title", "author", "address", "email", "thanks", "date", "keywords", "subjclass", "affil", "curraddr", "urladdr", "dedicatory"):
        body = remove_cmd(body, [c])
    full_body = rest.split("\\end{document}")[0] if rest else flat
    return Doc(body, thm, proofs), body, full_body


def count_cases(t: str) -> int:
    pat = re.compile(
        r"(?:^|\n|\\item\s*\[|\\paragraph\*?\{|\\(?:textbf|textit|emph|textsc|underline)\{|\{\\(?:bf|it|em)\s+|\.\s+)\s*(?:Sub)?[Cc]ase\s*(?:[0-9]+[a-z]?(?:\.[0-9]+)*|[IVX]+|\([0-9a-z]+\)|[A-Z]\b|[a-z]\)|\\ref|\$[^$]{1,20}\$|\\\()",
    )
    return len(pat.findall(t))


def count_display(t: str) -> int:
    return len(DISPLAY_ENVS.findall(t)) + len(re.findall(r"\\\[", t)) + len(re.findall(r"\$\$", t)) // 2


def extract(it: dict, root: Path, authors: list[str]) -> dict:
    doc, body, full_body = load_doc(root)
    loc = locate(doc, it)
    reg = build_region(doc, loc, it)
    R = Renderer(doc, authors)
    spans = reg["spans"]
    pieces = [(a, b, R.render(a, b)) for a, b, _, _ in spans]
    full_text = "\n\n[...]\n\n".join(p[2] for p in pieces)
    full_text_n = normalize(full_text)
    region_latex = "".join(doc.s[a:b] for a, b, _, _ in spans)
    truncated = len(full_text_n) > CAP
    if not truncated:
        text = full_text_n
    else:
        text = truncate(doc, loc, reg, R)
    text, removed = R.blind(text)
    text = normalize(text)
    full_text_b, _ = R.blind(full_text_n)
    main = loc["main"]
    sts = [e for e in doc.statements() if any(a <= e["start"] < b for a, b, _, _ in spans) and e is not main]
    n_lemmas = sum(1 for e in sts if RESULT_KINDS.search(e["name"]))
    cite_keys = set()
    for m in CITE_RE.finditer(full_body):
        cite_keys.update(k.strip() for k in m.group(1).split(",") if k.strip())
    return dict(
        text=text, removed=removed, method=loc["method"], main_name=main["name"], main_opt=main["opt"],
        main_head=normalize(R.render(main["start"], main["end"]))[:300],
        full_paper_chars=len(body), full_proof_chars=len(normalize(full_text_b)), proof_chars=len(text), truncated=truncated,
        n_lemmas=n_lemmas, n_cases_estimate=count_cases(region_latex), n_display_equations=count_display(region_latex),
        cites_count=len(cite_keys), n_deps=len(reg["deps"]), region_sections=[x["title"][:60] for x in reg["region_secs"]],
    )


def truncate(doc: Doc, loc: dict, reg: dict, R: Renderer) -> str:
    """Priority-fill the proof region up to CAP characters; output in document order."""
    s = doc.s
    main, proof, psec = loc["main"], loc["proof"], loc["proof_section"]
    region = reg["spans"]

    def inside(p: int) -> bool:
        return any(a <= p < b for a, b, _, _ in region)

    items: list[tuple[int, int, int, str]] = []  # (priority, start, end, kind)
    items.append((0, main["start"], main["end"], "main"))
    if proof is not None:
        items.append((1, proof["start"], proof["end"], "mainproof"))
    elif psec is not None:
        items.append((1, psec["start"], psec["end"], "mainproof"))
    for a, b, role, _ in region:
        if role == "overview":
            items.append((2, a, b, "overview"))
    for sec in doc.sections:
        if inside(sec["start"]):
            items.append((3, sec["start"], sec["head_end"], "heading"))
    deps = sorted(reg["deps"], key=lambda e: (reg["depth"][id(e)], e["start"]))
    for e in deps:
        if inside(e["start"]):
            items.append((10 + reg["depth"][id(e)], e["start"], e["end"], "dep_stmt"))
    others = [e for e in doc.statements() if inside(e["start"]) and e is not main and id(e) not in reg["depth"] and e["depth"] == 0]
    for e in others:
        items.append((40, e["start"], e["end"], "stmt"))
    for e in deps:
        pr = doc.proof_of(e)
        if pr and inside(pr["start"]):
            items.append((50 + reg["depth"][id(e)], pr["start"], pr["end"], "dep_proof"))
    for d, sec in reg["sec_proofs"]:
        if inside(sec["start"]):
            items.append((50 + d, sec["start"], sec["end"], "dep_proof"))
    for e in others:
        pr = doc.proof_of(e)
        if pr:
            items.append((90, pr["start"], pr["end"], "proof"))
    # last: remaining prose of the region, paragraph by paragraph, in document order
    for a, b, role, _ in region:
        last = a
        for m in re.finditer(r"\n[ \t]*\n", s[a:b]):
            p = a + m.start()
            seg = s[last:p]
            if len(re.findall(r"\\begin\{", seg)) == len(re.findall(r"\\end\{", seg)) and seg.strip():
                items.append((95, last, p, "para"))
                last = p
        if s[last:b].strip():
            items.append((95, last, b, "para"))
    items.sort(key=lambda x: (x[0], x[1]))
    budget = CAP - 400  # headroom for markers and blinding substitutions
    mainspan = (main["start"], main["end"])

    def para_cut(pos: int, lo: int, hi: int) -> int:
        """Nearest blank line to pos (inside [lo, hi]) that is not inside a display environment."""
        best = pos
        for m in re.finditer(r"\n\s*\n", s[lo:hi]):
            p = lo + m.start()
            pre = s[lo:p]
            if len(re.findall(r"\\begin\{", pre)) != len(re.findall(r"\\end\{", pre)):
                continue
            if abs(p - pos) < abs(best - pos) or best == pos:
                best = p
        return best

    def assemble(spans: list[tuple[int, int]]) -> str:
        R.cites = {}
        merged: list[tuple[int, int]] = []
        for a, b in sorted(set(spans)):
            if (a, b) == mainspan:
                continue
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        out = [R.render(*mainspan)]
        for a, b in merged:
            out.append("[...]")
            out.append(R.render(a, b))
        out.append("[...]")
        return normalize("\n\n".join(out))

    chosen: list[tuple[int, int]] = [mainspan]
    used = len(normalize(R.render(*mainspan)))
    for pr, a, b, kind in items:
        if kind == "main":
            continue
        if any(x <= a and b <= y for x, y in chosen):
            continue
        L = len(normalize(R.render(a, b))) + 8
        if kind == "mainproof" and used + L > budget:
            # keep the head and the tail of an over-long main proof, cut at paragraph boundaries
            keep = max(1500, budget - used - 3000)
            frac = keep / max(1, L)
            c1 = para_cut(a + int(0.75 * frac * (b - a)), a, b)
            c2 = para_cut(b - int(0.25 * frac * (b - a)), c1, b)
            for span in ((a, c1), (c2, b)):
                if span[1] > span[0]:
                    chosen.append(span)
                    used += len(normalize(R.render(*span))) + 8
            continue
        if used + L > budget:
            continue
        chosen.append((a, b))
        used += L
    text = assemble(chosen)
    while len(text) > budget and len(chosen) > 1:
        chosen.pop()
        text = assemble(chosen)
    return text


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------


def opaque_id(salt: str, key: str) -> str:
    return "p" + hashlib.sha256((salt + "|" + key).encode()).hexdigest()[:8]


def build(sel: dict, only: set | None = None, verbose: bool = False) -> None:
    salt = sel["id_salt"]
    rows = []
    audit = []
    for side in ("ai", "human"):
        (DATA / side).mkdir(parents=True, exist_ok=True)
        if only is None:
            for f in (DATA / side).glob("*.txt"):
                f.unlink()
        for it in sel[side]:
            key = it["slug"] if side == "ai" else it["arxiv_id"]
            if only and key not in only:
                continue
            pid = opaque_id(salt, side + ":" + key)
            if side == "ai":
                root = RAW / "ai" / it["slug"]
                authors: list[str] = []
            else:
                root = RAW / "human" / it["arxiv_version"]
                if not root.exists():
                    unpack_eprint(RAW / "human" / f"{it['arxiv_version']}.eprint", root)
                authors = it.get("authors", [])
            r = extract(it, root, authors)
            leaks = blinding_leaks(r["text"], authors)
            if leaks:
                raise RuntimeError(f"blinding leak in {side} {key}: {leaks}")
            path = DATA / side / f"{pid}.txt"
            path.write_text(r["text"])
            row = dict(id=pid, source=side, role=it.get("role", "primary"), field=it["field"], kind=it.get("kind"), title=it["title"],
                       problem_name=it["problem_name"], url=it["url"])
            if side == "ai":
                row.update(has_lean=True, lean_status=it["lean_status"], lean_declaration=it["lean_declaration"], lean_file=it["lean_file"],
                           lean_path=it["lean_file"], comparator_config=it["comparator_config"], family=it["family"])
            else:
                row.update(year=it["year"], arxiv_id=it["arxiv_id"], arxiv_version=it["arxiv_version"], doi=it.get("doi"), journal=it.get("journal"),
                           pub_year=it.get("pub_year"), cited_by=it.get("cited_by"), acceptance_evidence=it.get("acceptance_evidence"))
            row.update({k: r[k] for k in ("full_paper_chars", "full_proof_chars", "proof_chars", "truncated", "n_lemmas", "n_cases_estimate", "n_display_equations", "cites_count")})
            row["extraction_method"] = r["method"]
            row["path"] = str(path.relative_to(HERE))
            rows.append(row)
            audit.append(dict(id=pid, key=key, side=side, method=r["method"], main_head=r["main_head"], removed=r["removed"], region_sections=r["region_sections"], n_deps=r["n_deps"],
                              author_tokens=r["text"].count("[author]")))
            if verbose:
                print(f"{side:5s} {key[:40]:40s} {r['method']:20s} full={r['full_proof_chars']:7d} out={r['proof_chars']:6d} | {r['main_head'][:70]!r}")
    if only is None:
        with open(DATA / "corpus.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        (RAW / "extraction_audit.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False))
        write_report(sel, rows, audit)
        print(f"wrote {len(rows)} rows -> data/corpus.jsonl, data/corpus_report.md")


LEAK_RE = re.compile(r"OpenAI|\bGPT\b|ChatGPT|model-generated|AI-generated|\bLean\b|[Mm]athlib|arXiv:\s*\d{4}\.\d{4}")


def blinding_leaks(text: str, authors: list[str]) -> list[str]:
    leaks = LEAK_RE.findall(text)
    for full in authors:
        sur = full.split()[-1]
        if len(sur) >= 3 and re.search(r"(?<![A-Za-z\\])" + re.escape(sur) + r"(?![a-z])", PROTECT.sub("", text)):
            leaks.append(sur)
    return leaks


def write_report(sel: dict, rows: list[dict], audit: list[dict]) -> None:
    import statistics as st

    fields = sorted({r["field"] for r in rows})

    def count(side, role=None, field=None, kind=None):
        return sum(1 for r in rows if r["source"] == side and (role is None or r["role"] == role) and (field is None or r["field"] == field) and (kind is None or r["kind"] == kind))

    L = ["# Proof corpus report", "", "Generated by `build_corpus.py` from `data/selection.json` and `data/raw/`. Do not edit by hand.", ""]
    L += [f"- AI items: {count('ai', 'primary')} primary + {count('ai', 'reserve')} reserve",
          f"- Human items: {count('human', 'primary')} primary + {count('human', 'reserve')} reserve",
          f"- Proof-text cap: {CAP:,} characters (longer proofs truncated by priority; `truncated`, `full_proof_chars` record it)", ""]
    L += ["## Field table (primary set; reserves in parentheses)", "", "| Field | AI | Human | diff |", "|---|---:|---:|---:|"]
    for f in fields:
        a, h = count("ai", "primary", f), count("human", "primary", f)
        L.append(f"| {f} | {a} ({count('ai', 'reserve', f)}) | {h} ({count('human', 'reserve', f)}) | {a - h:+d} |")
    L.append(f"| **Total** | {count('ai', 'primary')} | {count('human', 'primary')} | |")
    L += ["", "## Result kind (primary set)", "", "| Kind | AI | Human |", "|---|---:|---:|"]
    for k in ("proof", "counterexample"):
        L.append(f"| {k} | {count('ai', 'primary', kind=k)} | {count('human', 'primary', kind=k)} |")
    L += ["", "## Length and structure (primary set, medians)", "", "| Metric | AI | Human |", "|---|---:|---:|"]
    for m in ("full_paper_chars", "full_proof_chars", "proof_chars", "n_lemmas", "n_cases_estimate", "n_display_equations", "cites_count"):
        vals = {s: [r[m] for r in rows if r["source"] == s and r["role"] == "primary"] for s in ("ai", "human")}
        L.append(f"| {m} | {st.median(vals['ai']):,.0f} | {st.median(vals['human']):,.0f} |")
    tr = {s: sum(r["truncated"] for r in rows if r["source"] == s and r["role"] == "primary") for s in ("ai", "human")}
    L.append(f"| truncated (count) | {tr['ai']} | {tr['human']} |")
    meth = {s: dict(sorted(collections.Counter(r["extraction_method"] for r in rows if r["source"] == s).items())) for s in ("ai", "human")}
    L += ["", f"Extraction method counts: AI {meth['ai']}; human {meth['human']}. `none` means no single main-proof block was found and the whole proof region was used.", ""]
    for sec in sel.get("report_sections", []):
        L += [f"## {sec['title']}", ""] + sec["lines"] + [""]
    L += ["## Gates", "", "| Gate | Candidates removed |", "|---|---:|"]
    for g in sel.get("gates", []):
        L.append(f"| {g['gate']} | {g['removed']} |")
    L += ["", "## Exclusions", "", "| Side | Item | Gate | Reason |", "|---|---|---|---|"]
    for x in sel.get("exclusions", []):
        L.append(f"| {x['side']} | {x['item']} | {x['gate']} | {x['reason']} |")
    tok = [a for a in audit if a["author_tokens"]]
    L += ["", "## Blinding audit", "",
          f"- Sentences removed by the blinding filters: {sum(len(a['removed']) for a in audit)} (AI {sum(len(a['removed']) for a in audit if a['side'] == 'ai')}, human {sum(len(a['removed']) for a in audit if a['side'] == 'human')}); full list in `data/raw/extraction_audit.json`.",
          f"- `[author]` placeholders left inside mathematical sentences or headings (eponyms such as a self-cited theorem name): {sum(a['author_tokens'] for a in tok)} in {len(tok)} human texts.",
          "- The build fails if any output text contains OpenAI/GPT/Lean/mathlib/arXiv-id strings or an author surname of its own paper.", ""]
    (DATA / "corpus_report.md").write_text("\n".join(L) + "\n")


def main() -> None:
    sel = json.loads((DATA / "selection.json").read_text())
    args = sys.argv[1:]
    if args[:1] == ["fetch"]:
        fetch(sel)
        return
    only = set(args[1:]) if args[:1] == ["only"] else None
    build(sel, only=only, verbose=True)


if __name__ == "__main__":
    main()
