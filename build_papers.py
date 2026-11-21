"""Build publication-ready PDFs, arXiv LaTeX bundles, and threebits.si/papers/ pages.

Reads the three canonical Markdown manuscripts:
  1. paper_causal_brain_sim.md
  2. paper_falcon_lowbit.md
  3. paper_prosthetic_cortex.md

Outputs:
  - papers/threebits_paper{1,2,3}_*.pdf                (bioRxiv / Zenodo ready PDFs)
  - papers/threebits_paper{1,2,3}_*_arxiv.tar.gz       (arXiv / Overleaf LaTeX bundles)
  - papers/arxiv/paper{1,2,3}_*/main.tex + figures/    (unpacked LaTeX source)
  - website/papers/{index.html, *.html, pdfs/, figures/, arxiv/}
"""
import html
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPERS_DIR = ROOT / "papers"
WEB_PAPERS = ROOT / "website" / "papers"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

SPECS = [
    {
        "num": 1,
        "slug": "causal-brain-sim",
        "file_stem": "threebits_paper1_causal_brain_sim",
        "md": ROOT / "paper_causal_brain_sim.md",
        "venue_hint": "Target: bioRxiv (Neuroscience) · PLOS Computational Biology / Imaging Neuroscience",
        "tag": "Whole-Brain Causal Simulation & Safe Control",
        "highlights": [
            "74 CCEP patients (316,366 trials): resting-fMRI tuning drops causal ρ from +0.62 → +0.18",
            "Multi-scale 19-receptor PET + MEG + Mesulam twin jumps unseen-region partial ρ (+0.319 → +0.366) & LOO-network (+0.374), beating permuted controls (+0.321) and reaching ρ = +0.716 on unseen pairs",
            "Zero-shot transfer to 44/46 healthy TMS-EEG adults + 160/160 Lyapunov BRL certificates + 15-patient RAM memory policy",
        ],
        "bibkey": "tahir2026causal",
        "bibtex_title": "Observational Resting-State Tuning Degrades Causal Electrical-Stimulation Accuracy in Human Whole-Brain Models",
    },
    {
        "num": 2,
        "slug": "falcon-lowbit",
        "file_stem": "threebits_paper2_falcon_lowbit",
        "md": ROOT / "paper_falcon_lowbit.md",
        "venue_hint": "Target: bioRxiv / arXiv (q-bio.NC, cs.NE) · Journal of Neural Engineering",
        "tag": "Ultra-Compact Sub-8-Bit Neural Implant Decoders",
        "highlights": [
            "3-bit (44.1 kB) & 4-bit (58.8 kB) causal GRU match 32-bit R² = 0.816 on human FALCON H1",
            "Unsupervised mean-align + 48 s L2-anchored 1% adapter handles 6-week drift with 99% core read-only",
            "Pure-integer C99 int4×int8 kernel + RP2350 header: 66.35 kB total ROM+SRAM, 14.3 µs/bin, 0.8191 R²",
        ],
        "bibkey": "tahir2026lowbit",
        "bibtex_title": "Sub-8-Bit Quantized Neural Decoders Under Multi-Week Human Intracortical Drift",
    },
    {
        "num": 3,
        "slug": "prosthetic-cortex",
        "file_stem": "threebits_paper3_prosthetic_cortex",
        "md": ROOT / "paper_prosthetic_cortex.md",
        "venue_hint": "Target: bioRxiv / arXiv (q-bio.NC, cs.AI) · Journal of Neural Engineering",
        "tag": "Human Working-Memory Limits & Brain–AI Phase Transition",
        "highlights": [
            "123 healthy adults (ds008037): empirical Cowan cliff 88.6% → 43.8% (K=2→6); pre-response EEG overload 0.690 AUC",
            "2D Bandwidth × Latency phase transition at K=12: 320 bps (~265 bps MI) at ≤100 ms jumps 33% → 91.5% (97.4% at K=16) & Level-4 Strict Outperformance (+63.7 pp)",
            "Triple-validated 100-region Functional Human Brain Emulator (post-train CCEP ρ=+0.66, 45/46 TMS-EEG +0.29): Hippocampal Index (100% through K=8, 87.2% K=12) & 4-bit Region 101 (72.1% vs 35.6% Sham) break Cowan's wall",
        ],
        "bibkey": "tahir2026prosthetic",
        "bibtex_title": "From Biological Working-Memory Limits to Co-Adaptive Brain--AI Co-Processing: A Quantitative Proof Ladder",
    },
]


# ------------------------- Math & Markdown -> Clean HTML -------------------------
def tex_to_html(expr: str) -> str:
    """Convert inline/display LaTeX snippets in our 3 manuscripts into crisp HTML math."""
    s = expr.strip()
    # Special fractions & nested exponents first
    s = re.sub(r"\\frac\{dz_j\}\{dt\}", "<i>dz</i><sub>j</sub>/<i>dt</i>", s)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", s)
    s = re.sub(r"\\sum_\{([^{}]+)\}\^\{?([^{}\s]+)\}?", r"∑<sub>\1</sub><sup>\2</sup>", s)
    s = re.sub(r"\\operatorname\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"\\text\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{1\}", "<b>1</b>", s)
    s = re.sub(r"\\mathbb\{R\}", "ℝ", s)
    s = s.replace(r"e^{g_{ij}}", "<i>e</i><sup>g<sub>ij</sub></sup>")
    repls = [
        (r"\big(", "("), (r"\big)", ")"), (r"\left", ""), (r"\right", ""),
        (r"\log", "log"), (r"\tanh", "tanh"), (r"\max", "max"), (r"\min", "min"),
        (r"\rho", "ρ"), (r"\alpha", "α"), (r"\gamma", "γ"), (r"\omega", "ω"),
        (r"\sigma", "σ"), (r"\eta", "η"), (r"\lambda", "λ"), (r"\Delta", "Δ"),
        (r"\tau", "τ"), (r"\theta", "θ"), (r"\pi", "π"), (r"\infty", "∞"),
        (r"\to", "→"), (r"\mapsto", "↦"), (r"\mid", " | "), (r"\le", "≤"), (r"\ge", "≥"),
        (r"\ne", "≠"), (r"\sim", "~"), (r"\times", "×"), (r"\pm", "±"), (r"\propto", "∝"),
        (r"\prec", "≺"), (r"\succ", "≻"), (r"\forall", "∀"), (r"\in", "∈"), (r"\dots", "…"),
        (r"\circ", "°"), (r"\|", "‖"), (r"\;", " "), (r"\,", " "), (r"\ ", " "),
    ]
    for a, b in repls:
        s = s.replace(a, b)
    # Superscripts & subscripts (including Unicode like ∞, °)
    s = re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_\{([^{}]+)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"\^([0-9a-zA-Z°∞+\-]+)", r"<sup>\1</sup>", s)
    s = re.sub(r"_([0-9a-zA-Z∞]+)", r"<sub>\1</sub>", s)
    s = s.replace("{", "").replace("}", "")
    return f'<span class="math">{s}</span>'


def inline_md_to_html(text: str) -> str:
    # Protect inline code first
    tokens = []
    def hold(val):
        idx = len(tokens)
        tokens.append(val)
        return f"\x00TOK{idx}\x00"

    text = re.sub(r"`([^`]+)`", lambda m: hold(f"<code>{html.escape(m.group(1))}</code>"), text)
    text = re.sub(r"\$([^$]+)\$", lambda m: hold(tex_to_html(m.group(1))), text)
    text = html.escape(text)
    # Links (bare https://...)
    text = re.sub(r"(https?://[^\s<)]+)", lambda m: f'<a href="{m.group(1)}" target="_blank" rel="noopener">{m.group(1)}</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    for idx, val in enumerate(tokens):
        text = text.replace(f"\x00TOK{idx}\x00", val)
    return text


def md_to_html_body(md_text: str, fig_prefix: str):
    lines = md_text.splitlines()
    title = ""
    meta_lines = []
    i = 0
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        i = 1
    while i < len(lines) and lines[i].strip() != "---":
        if lines[i].strip():
            meta_lines.append(lines[i].strip())
        i += 1
    if i < len(lines) and lines[i].strip() == "---":
        i += 1

    out = []
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    while i < len(lines):
        ln = lines[i].rstrip()
        s = ln.strip()
        if not s:
            close_lists()
            i += 1
            continue
        if s == "---":
            close_lists()
            out.append("<hr>")
            i += 1
            continue
        if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            close_lists()
            out.append(f'<div class="eq">{tex_to_html(s[2:-2])}</div>')
            i += 1
            continue
        if s.startswith("## "):
            close_lists()
            out.append(f"<h2>{inline_md_to_html(s[3:])}</h2>")
            i += 1
            continue
        if s.startswith("### "):
            close_lists()
            out.append(f"<h3>{inline_md_to_html(s[4:])}</h3>")
            i += 1
            continue
        # Image
        m_img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
        if m_img:
            close_lists()
            cap, src = m_img.group(1), m_img.group(2)
            fname = Path(src).name
            out.append(
                f'<figure class="paper-fig">'
                f'<img src="{fig_prefix}{fname}" alt="{html.escape(cap)}">'
                f'<figcaption>{inline_md_to_html(cap)}</figcaption>'
                f"</figure>"
            )
            i += 1
            continue
        # Table
        if s.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            close_lists()
            headers = [c.strip() for c in s.strip("|").split("|")]
            i += 2  # skip separator
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            th_html = "".join(f"<th>{inline_md_to_html(h)}</th>" for h in headers)
            body_html = ""
            for r in rows:
                tds = "".join(f"<td>{inline_md_to_html(c)}</td>" for c in r)
                body_html += f"<tr>{tds}</tr>"
            out.append(f'<div class="tbl-wrap"><table><thead><tr>{th_html}</tr></thead><tbody>{body_html}</tbody></table></div>')
            continue
        # Bullet list
        if s.startswith("- "):
            if in_ol:
                out.append("</ol>")
                in_ol = False
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{inline_md_to_html(s[2:])}</li>")
            i += 1
            continue
        # Numbered list
        m_ol = re.match(r"^\d+\.\s+(.*)$", s)
        if m_ol:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            out.append(f"<li>{inline_md_to_html(m_ol.group(1))}</li>")
            i += 1
            continue
        # Paragraph
        close_lists()
        para = [s]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith(("#", "- ", "|", "![", "$$", "---")) or re.match(r"^\d+\.\s+", nxt):
                break
            para.append(nxt)
            i += 1
        out.append(f"<p>{inline_md_to_html(' '.join(para))}</p>")

    close_lists()
    return title, meta_lines, "\n".join(out)


# ------------------------- Markdown -> Clean arXiv LaTeX -------------------------
def escape_tex_text(s: str) -> str:
    """Escape plain prose for pdflatex (preserving already-protected tokens)."""
    # Replace Unicode first
    uni = {
        "ρ": r"$\rho$", "α": r"$\alpha$", "γ": r"$\gamma$", "ω": r"$\omega$",
        "σ": r"$\sigma$", "η": r"$\eta$", "λ": r"$\lambda$", "Δ": r"$\Delta$",
        "τ": r"$\tau$", "θ": r"$\theta$", "≤": r"$\le$", "≥": r"$\ge$",
        "≠": r"$\ne$", "×": r"$\times$", "±": r"$\pm$", "→": r"$\to$",
        "−": r"$-$", "°": r"$^\circ$", "µ": r"$\mu$", "²": r"$^2$",
        "∈": r"$\in$", "–": "--", "—": "---", "·": r"$\cdot$", "…": r"\ldots{}",
    }
    out = []
    for ch in s:
        if ch in uni:
            out.append(uni[ch])
        elif ch == "%":
            out.append(r"\%")
        elif ch == "&":
            out.append(r"\&")
        elif ch == "_":
            out.append(r"\_")
        elif ch == "#":
            out.append(r"\#")
        elif ch == "$":
            out.append(r"\$")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        else:
            out.append(ch)
    return "".join(out)


def inline_md_to_tex(text: str) -> str:
    tokens = []
    def hold(val):
        idx = len(tokens)
        tokens.append(val)
        return f"XXTOK{idx}XX"

    # Protect math spans ($...$)
    def fix_math(m):
        inner = m.group(1)
        for u, r in [("−", "-"), ("°", r"^\circ"), ("ρ", r"\rho "), ("Δ", r"\Delta "), ("≤", r"\le "), ("≥", r"\ge "), ("×", r"\times "), ("→", r"\to "), ("±", r"\pm "), ("µ", r"\mu "), ("∈", r"\in "), ("–", "-")]:
            inner = inner.replace(u, r)
        return hold("$" + inner + "$")

    text = re.sub(r"\$([^$]+)\$", fix_math, text)
    text = re.sub(r"`([^`]+)`", lambda m: hold(r"\texttt{" + escape_tex_text(m.group(1)) + "}"), text)
    text = re.sub(r"(https?://[^\s)]+)", lambda m: hold(r"\url{" + m.group(1) + "}"), text)
    text = re.sub(r"\*\*([^*]+)\*\*", lambda m: hold(r"\textbf{" + escape_tex_text(m.group(1)) + "}"), text)
    text = re.sub(r"\*([^*]+)\*", lambda m: hold(r"\emph{" + escape_tex_text(m.group(1)) + "}"), text)
    text = escape_tex_text(text)
    for idx in range(len(tokens) - 1, -1, -1):
        text = text.replace(f"XXTOK{idx}XX", tokens[idx])
    return text


def md_to_arxiv_tex(md_text: str):
    lines = md_text.splitlines()
    title = lines[0][2:].strip()
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        i += 1
    if i < len(lines):
        i += 1

    body = []
    figs_used = []
    in_ul = False
    in_ol = False
    in_abs = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            body.append(r"\end{itemize}")
            in_ul = False
        if in_ol:
            body.append(r"\end{enumerate}")
            in_ol = False

    while i < len(lines):
        s = lines[i].strip()
        if not s:
            close_lists()
            i += 1
            continue
        if s == "---":
            close_lists()
            if in_abs:
                body.append(r"\end{abstract}")
                in_abs = False
            i += 1
            continue
        if s.startswith("$$") and s.endswith("$$"):
            close_lists()
            body.append(r"\begin{equation*}" + s[2:-2] + r"\end{equation*}")
            i += 1
            continue
        if s.startswith("## "):
            close_lists()
            if in_abs:
                body.append(r"\end{abstract}")
                in_abs = False
            hdr = s[3:].strip()
            if hdr.lower() == "abstract":
                body.append(r"\begin{abstract}")
                in_abs = True
            else:
                hdr_clean = re.sub(r"^\d+\.\s*", "", hdr)
                if hdr.lower() in ("references", "ethics statement, data & code availability, and competing interests"):
                    body.append(r"\section*{" + inline_md_to_tex(hdr_clean) + "}")
                else:
                    body.append(r"\section{" + inline_md_to_tex(hdr_clean) + "}")
            i += 1
            continue
        if s.startswith("### "):
            close_lists()
            hdr = re.sub(r"^\d+\.\d+\s*", "", s[4:].strip())
            body.append(r"\subsection{" + inline_md_to_tex(hdr) + "}")
            i += 1
            continue
        m_img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
        if m_img:
            close_lists()
            cap, src = m_img.group(1), m_img.group(2)
            cap = re.sub(r"^Figure\s+\d+:\s*", "", cap)
            fname = Path(src).name
            figs_used.append(fname)
            body.append(
                "\\begin{figure}[htbp]\n\\centering\n"
                f"\\includegraphics[width=\\linewidth]{{figures/{fname}}}\n"
                f"\\caption{{{inline_md_to_tex(cap)}}}\n"
                "\\end{figure}"
            )
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            close_lists()
            headers = [c.strip() for c in s.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            ncols = len(headers)
            colspec = "l" + "c" * (ncols - 1)
            body.append("\\begin{table}[htbp]\n\\centering\n\\small\n\\resizebox{\\linewidth}{!}{%")
            body.append(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule")
            body.append(" & ".join(inline_md_to_tex(h) for h in headers) + r" \\ \midrule")
            for r in rows:
                body.append(" & ".join(inline_md_to_tex(c) for c in r) + r" \\")
            body.append("\\bottomrule\n\\end{tabular}}\n\\end{table}")
            continue
        if s.startswith("- "):
            if in_ol:
                body.append(r"\end{enumerate}")
                in_ol = False
            if not in_ul:
                body.append(r"\begin{itemize}")
                in_ul = True
            body.append(r"\item " + inline_md_to_tex(s[2:]))
            i += 1
            continue
        m_ol = re.match(r"^\d+\.\s+(.*)$", s)
        if m_ol:
            if in_ul:
                body.append(r"\end{itemize}")
                in_ul = False
            if not in_ol:
                body.append(r"\begin{enumerate}")
                in_ol = True
            body.append(r"\item " + inline_md_to_tex(m_ol.group(1)))
            i += 1
            continue
        close_lists()
        para = [s]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith(("#", "- ", "|", "![", "$$", "---")) or re.match(r"^\d+\.\s+", nxt):
                break
            para.append(nxt)
            i += 1
        body.append(inline_md_to_tex(" ".join(para)) + "\n")

    close_lists()
    if in_abs:
        body.append(r"\end{abstract}")

    tex = f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage[margin=0.9in]{{geometry}}
\\usepackage{{amsmath,amssymb,booktabs,graphicx,microtype}}
\\usepackage[hidelinks]{{hyperref}}
\\usepackage{{url}}

\\title{{{inline_md_to_tex(title)}}}
\\author{{Muhammad Farhan Tahir\\\\Threebits Research\\\\\\url{{https://threebits.si}}}}
\\date{{21 November 2026}}

\\begin{{document}}
\\maketitle

{chr(10).join(body)}

\\end{{document}}
"""
    return tex, figs_used


# ------------------------- HTML Templates (Print PDF + Live Web) -------------------------
PRINT_CSS = """
@page { size: letter; margin: 0.78in 0.78in 0.85in 0.78in; }
body { font-family: Georgia, "Times New Roman", serif; color: #161513; font-size: 10.3pt; line-height: 1.54; margin: 0; }
.paper-hdr { border-bottom: 1.5px solid #161513; padding-bottom: 12pt; margin-bottom: 14pt; }
.preprint-kicker { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 8pt; letter-spacing: 0.1em; text-transform: uppercase; color: #b5431f; font-weight: 600; margin-bottom: 6pt; }
h1 { font-size: 18pt; line-height: 1.18; font-weight: 700; margin: 0 0 8pt; letter-spacing: -0.015em; }
.authors { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 9.5pt; color: #3a3731; line-height: 1.45; }
h2 { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 12pt; font-weight: 700; margin: 16pt 0 6pt; color: #161513; border-bottom: 0.5px solid #d8d4c9; padding-bottom: 2.5pt; break-after: avoid; }
h3 { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 10.4pt; font-weight: 600; margin: 11pt 0 4pt; color: #26241f; break-after: avoid; }
p { margin: 0 0 7.5pt; text-align: justify; hyphens: auto; }
ul, ol { margin: 4pt 0 8pt 18pt; padding: 0; }
li { margin-bottom: 4pt; }
hr { border: none; border-top: 0.5px solid #d8d4c9; margin: 12pt 0; }
code { font-family: "IBM Plex Mono", Menlo, Consolas, monospace; font-size: 8.6pt; background: #f3f1eb; padding: 0.5pt 3pt; border-radius: 2pt; }
.math { font-family: Georgia, "Times New Roman", serif; }
.eq { text-align: center; font-size: 10.6pt; margin: 9pt 0; padding: 6pt; background: #faf9f5; border-left: 2px solid #b5431f; break-inside: avoid; }
.tbl-wrap { margin: 9pt 0 11pt; break-inside: avoid; }
table { width: 100%; border-collapse: collapse; font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 8.1pt; line-height: 1.32; }
th, td { padding: 4.5pt 5pt; border-bottom: 0.5px solid #dcd8cd; text-align: left; vertical-align: top; }
thead th { border-top: 1.2px solid #161513; border-bottom: 1px solid #161513; font-weight: 600; background: #f7f5ef; }
.paper-fig { margin: 11pt 0 13pt; text-align: center; break-inside: avoid; }
.paper-fig img { max-width: 100%; height: auto; border: 0.5px solid #dcd8cd; border-radius: 3pt; }
.paper-fig figcaption { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 8.4pt; color: #47443c; text-align: left; margin-top: 5pt; line-height: 1.4; }
a { color: #b5431f; text-decoration: none; }
"""

WEB_PAPER_CSS = """
:root { --bg:#f9f8f4; --panel:#f0eee7; --surface:#ffffff; --ink:#1a1916; --ink2:#47443c; --muted:#6b675b; --line:#e4e1d7; --accent:#b5431f; --decoded:#c8502a;
  --display:"Newsreader",Georgia,serif; --body:"Inter",-apple-system,sans-serif; --mono:"IBM Plex Mono",ui-monospace,monospace; }
@media (prefers-color-scheme:dark){ :root { --bg:#151411; --panel:#1e1d19; --surface:#24221d; --ink:#efece3; --ink2:#bdb8aa; --muted:#948f81; --line:#2e2c27; --accent:#f0905f; --decoded:#f07f4e; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:400 1.03rem/1.7 var(--body); -webkit-font-smoothing:antialiased; }
a { color:var(--accent); text-underline-offset:0.2em; }
.top { position:sticky; top:0; z-index:20; border-bottom:1px solid var(--line); background:color-mix(in srgb, var(--bg) 92%, transparent); backdrop-filter:blur(12px); }
.bar { max-width:68rem; margin:0 auto; padding:0 24px; min-height:3.75rem; display:flex; align-items:center; justify-content:space-between; gap:16px; }
.mark { display:inline-flex; align-items:center; gap:9px; font-weight:600; color:var(--ink); text-decoration:none; }
.mark i { display:flex; gap:3px; } .mark i::before, .mark i::after, .mark i b { content:""; width:7px; height:7px; background:var(--ink); } .mark i::after { background:var(--decoded); }
.top nav { display:flex; flex-wrap:wrap; gap:18px; font-size:0.92rem; }
.top nav a { color:var(--ink2); text-decoration:none; } .top nav a:hover { color:var(--ink); }
.wrap { max-width:52rem; margin:0 auto; padding:42px 24px 72px; }
.kicker { font:500 0.75rem/1.3 var(--mono); letter-spacing:0.1em; text-transform:uppercase; color:var(--accent); margin-bottom:12px; }
h1 { font-family:var(--display); font-weight:400; font-size:clamp(2rem,4.5vw,3.1rem); line-height:1.08; letter-spacing:-0.025em; margin:0 0 14px; }
.meta { font-size:0.92rem; color:var(--ink2); margin-bottom:18px; }
.actions { display:flex; flex-wrap:wrap; gap:10px; margin:18px 0 30px; padding-bottom:24px; border-bottom:1px solid var(--line); }
.btn { display:inline-flex; align-items:center; gap:6px; padding:9px 16px; border-radius:999px; border:1px solid var(--ink); background:var(--ink); color:var(--bg); font-size:0.88rem; font-weight:500; text-decoration:none; cursor:pointer; font-family:var(--body); }
.btn:hover { background:var(--accent); border-color:var(--accent); color:#fff; }
.btn.ghost { background:transparent; color:var(--ink); border-color:var(--line); }
.btn.ghost:hover { border-color:var(--ink); background:transparent; color:var(--ink); }
h2 { font-family:var(--display); font-weight:400; font-size:1.75rem; line-height:1.15; margin:2.2rem 0 0.75rem; padding-top:0.5rem; border-top:1px solid var(--line); }
h3 { font-size:1.08rem; font-weight:600; margin:1.5rem 0 0.45rem; }
p { margin:0 0 1rem; color:var(--ink2); }
strong { color:var(--ink); }
ul, ol { margin:0 0 1rem 1.3rem; padding:0; color:var(--ink2); }
li { margin-bottom:0.45rem; }
hr { border:none; border-top:1px solid var(--line); margin:1.8rem 0; }
code { font-family:var(--mono); font-size:0.86em; background:var(--panel); padding:2px 5px; border-radius:4px; color:var(--ink); }
.eq { padding:14px 18px; margin:18px 0; background:var(--panel); border-left:3px solid var(--accent); border-radius:0 8px 8px 0; text-align:center; font-family:var(--display); font-size:1.15rem; overflow-x:auto; color:var(--ink); }
.tbl-wrap { overflow-x:auto; margin:18px 0 22px; border:1px solid var(--line); border-radius:10px; background:var(--surface); }
table { width:100%; border-collapse:collapse; font-size:0.84rem; line-height:1.4; }
th, td { padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; }
thead th { background:var(--panel); font-weight:600; color:var(--ink); }
.paper-fig { margin:24px 0; padding:14px; background:var(--surface); border:1px solid var(--line); border-radius:12px; }
.paper-fig img { width:100%; height:auto; display:block; border-radius:6px; background:#fff; }
.paper-fig figcaption { font-size:0.86rem; color:var(--muted); margin-top:10px; line-height:1.5; }
pre.bib { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; font:400 0.82rem/1.5 var(--mono); overflow-x:auto; color:var(--ink); }
"""

ICON_DATA = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' rx='3' fill='%231a1916'/%3E%3Crect x='3' y='3' width='4' height='4' fill='%23f9f8f4'/%3E%3Crect x='9' y='9' width='4' height='4' fill='%23ee6f40'/%3E%3C/svg%3E"


def make_bibtex(spec):
    return (
        f"@article{{{spec['bibkey']},\n"
        f"  author  = {{Tahir, Muhammad Farhan}},\n"
        f"  title   = {{{{{spec['bibtex_title']}}}}},\n"
        f"  journal = {{Threebits Research Preprint}},\n"
        f"  year    = {{2026}},\n"
        f"  month   = {{nov}},\n"
        f"  url     = {{https://threebits.si/papers/{spec['slug']}.html}}\n"
        f"}}"
    )


def render_pdf(print_html_path: Path, pdf_out_path: Path):
    with tempfile.TemporaryDirectory() as udir:
        cmd = [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            f"--user-data-dir={udir}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_out_path}",
            f"file://{print_html_path}",
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        assert cp.returncode == 0 and pdf_out_path.exists() and pdf_out_path.stat().st_size > 20000, cp.stderr


def build_all():
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    (PAPERS_DIR / "arxiv").mkdir(parents=True, exist_ok=True)
    for sub in ("pdfs", "figures", "arxiv"):
        (WEB_PAPERS / sub).mkdir(parents=True, exist_ok=True)

    # Copy all 5 figures into website/papers/figures/
    for fig in sorted((ROOT / "figures").glob("*.png")):
        shutil.copy2(fig, WEB_PAPERS / "figures" / fig.name)

    cards_html = []
    for spec in SPECS:
        md_text = spec["md"].read_text()
        title, meta_lines, print_body = md_to_html_body(md_text, fig_prefix=f"file://{ROOT / 'figures'}/")
        _, _, web_body = md_to_html_body(md_text, fig_prefix="figures/")
        # Verify zero unrendered LaTeX dollars or commands leaked into HTML
        assert "$" not in re.sub(r"\$\d", "", print_body), f"Unrendered $ found in {spec['slug']}"
        for bad in (r"\log", r"\tanh", r"\max", "_∞", "e^g"):
            assert bad not in print_body, f"Leaked LaTeX '{bad}' in {spec['slug']}"

        bib = make_bibtex(spec)
        # 1. Print PDF via Chrome
        print_html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{PRINT_CSS}</style></head><body>
<div class="paper-hdr">
  <div class="preprint-kicker">Threebits Research Preprint · Paper {spec['num']}</div>
  <h1>{html.escape(title)}</h1>
  <div class="authors">{'<br>'.join(inline_md_to_html(m) for m in meta_lines)}</div>
</div>
{print_body}
</body></html>"""
        with tempfile.TemporaryDirectory() as td:
            phtml = Path(td) / f"{spec['file_stem']}.html"
            phtml.write_text(print_html)
            pdf_path = PAPERS_DIR / f"{spec['file_stem']}.pdf"
            render_pdf(phtml, pdf_path)
            shutil.copy2(pdf_path, WEB_PAPERS / "pdfs" / pdf_path.name)

        # 2. arXiv LaTeX bundle (.tex + figures + .tar.gz)
        tex_src, figs_used = md_to_arxiv_tex(md_text)
        assert "XXTOK" not in tex_src, f"Leaked XXTOK in {spec['slug']} main.tex"
        ad = PAPERS_DIR / "arxiv" / spec["file_stem"]
        if ad.exists():
            shutil.rmtree(ad)
        (ad / "figures").mkdir(parents=True)
        (ad / "main.tex").write_text(tex_src)
        for fn in figs_used:
            shutil.copy2(ROOT / "figures" / fn, ad / "figures" / fn)
        tar_path = PAPERS_DIR / f"{spec['file_stem']}_arxiv.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(ad / "main.tex", arcname="main.tex")
            for fn in figs_used:
                tf.add(ad / "figures" / fn, arcname=f"figures/{fn}")
        shutil.copy2(tar_path, WEB_PAPERS / "arxiv" / tar_path.name)

        # 3. Online readable HTML page on threebits.si/papers/<slug>.html
        web_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="description" content="{html.escape(title)} — Threebits Research Preprint by Muhammad Farhan Tahir.">
<link rel="canonical" href="https://threebits.si/papers/{spec['slug']}.html">
<link rel="icon" href="{ICON_DATA}">
<title>{html.escape(title)} · Threebits</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..500;1,6..72,400..500&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{WEB_PAPER_CSS}</style>
</head>
<body>
<header class="top">
  <div class="bar">
    <a class="mark" href="/"><i aria-hidden="true"><b></b></i>Threebits</a>
    <nav>
      <a href="/papers/">&larr; All Papers</a>
      <a href="/gym/">Neural Gym</a>
      <a href="/">Research Overview</a>
    </nav>
  </div>
</header>
<main class="wrap">
  <div class="kicker">Threebits Preprint {spec['num']} · {html.escape(spec['tag'])}</div>
  <h1>{html.escape(title)}</h1>
  <div class="meta">{'<br>'.join(inline_md_to_html(m) for m in meta_lines)}</div>
  <div class="actions">
    <a class="btn" href="pdfs/{spec['file_stem']}.pdf" download>Download PDF</a>
    <a class="btn ghost" href="arxiv/{spec['file_stem']}_arxiv.tar.gz" download>arXiv LaTeX (.tar.gz)</a>
    <button class="btn ghost" id="copy-bib" type="button">Copy BibTeX</button>
  </div>
  {web_body}
  <h2>Citation (BibTeX)</h2>
  <pre class="bib" id="bib-box">{html.escape(bib)}</pre>
</main>
<script>
document.getElementById("copy-bib").onclick = async (e) => {{
  try {{
    await navigator.clipboard.writeText(document.getElementById("bib-box").textContent);
    e.target.textContent = "Copied BibTeX!";
    setTimeout(() => e.target.textContent = "Copy BibTeX", 1800);
  }} catch(_) {{}}
}};
</script>
</body>
</html>
"""
        (WEB_PAPERS / f"{spec['slug']}.html").write_text(web_html)

        # Card for /papers/index.html
        hi_li = "".join(f"<li>{html.escape(h)}</li>" for h in spec["highlights"])
        pdf_kb = round((PAPERS_DIR / f"{spec['file_stem']}.pdf").stat().st_size / 1024)
        cards_html.append(f"""
  <article class="paper-card">
    <div class="kicker">Preprint {spec['num']} · {html.escape(spec['tag'])}</div>
    <h2><a href="{spec['slug']}.html">{html.escape(title)}</a></h2>
    <p class="byline">Muhammad Farhan Tahir · Threebits Research · 21 November 2026 · <span>{html.escape(spec['venue_hint'])}</span></p>
    <ul class="hi">{hi_li}</ul>
    <div class="actions">
      <a class="btn" href="{spec['slug']}.html">Read Online</a>
      <a class="btn ghost" href="pdfs/{spec['file_stem']}.pdf" download>PDF ({pdf_kb} kB)</a>
      <a class="btn ghost" href="arxiv/{spec['file_stem']}_arxiv.tar.gz" download>LaTeX (.tar.gz)</a>
      <button class="btn ghost copy-btn" data-bib="{html.escape(bib)}" type="button">Copy BibTeX</button>
    </div>
  </article>""")
        print(f"Built Paper {spec['num']}: PDF={pdf_kb} kB | arXiv tar={round(tar_path.stat().st_size/1024)} kB | web={spec['slug']}.html")

    # 4. Hub index page: website/papers/index.html
    index_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="description" content="Threebits Research Preprints: causally validated human whole-brain simulation, certified-safe closed-loop control, sub-8-bit neural implant decoders, and Brain–AI co-processing.">
<link rel="canonical" href="https://threebits.si/papers/">
<link rel="icon" href="{ICON_DATA}">
<title>Research Papers · Threebits</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..500;1,6..72,400..500&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{ --bg:#f9f8f4; --panel:#f0eee7; --surface:#ffffff; --ink:#1a1916; --ink2:#47443c; --muted:#6b675b; --line:#e4e1d7; --accent:#b5431f; --decoded:#c8502a;
  --display:"Newsreader",Georgia,serif; --body:"Inter",-apple-system,sans-serif; --mono:"IBM Plex Mono",ui-monospace,monospace; }}
@media (prefers-color-scheme:dark){{ :root {{ --bg:#151411; --panel:#1e1d19; --surface:#24221d; --ink:#efece3; --ink2:#bdb8aa; --muted:#948f81; --line:#2e2c27; --accent:#f0905f; --decoded:#f07f4e; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:400 1.03rem/1.65 var(--body); -webkit-font-smoothing:antialiased; }}
a {{ color:var(--accent); text-underline-offset:0.2em; }}
.top {{ position:sticky; top:0; z-index:20; border-bottom:1px solid var(--line); background:color-mix(in srgb, var(--bg) 92%, transparent); backdrop-filter:blur(12px); }}
.bar {{ max-width:68rem; margin:0 auto; padding:0 24px; min-height:3.75rem; display:flex; align-items:center; justify-content:space-between; gap:16px; }}
.mark {{ display:inline-flex; align-items:center; gap:9px; font-weight:600; color:var(--ink); text-decoration:none; }}
.mark i {{ display:flex; gap:3px; }} .mark i::before, .mark i::after, .mark i b {{ content:""; width:7px; height:7px; background:var(--ink); }} .mark i::after {{ background:var(--decoded); }}
.top nav {{ display:flex; flex-wrap:wrap; gap:18px; font-size:0.92rem; }}
.top nav a {{ color:var(--ink2); text-decoration:none; }} .top nav a:hover {{ color:var(--ink); }}
.wrap {{ max-width:64rem; margin:0 auto; padding:48px 24px 72px; }}
.kicker {{ font:500 0.74rem/1.3 var(--mono); letter-spacing:0.1em; text-transform:uppercase; color:var(--accent); margin-bottom:8px; }}
h1 {{ font-family:var(--display); font-weight:400; font-size:clamp(2.2rem,5vw,3.6rem); line-height:1.06; letter-spacing:-0.025em; margin:0 0 14px; }}
h1 em {{ font-style:italic; color:var(--accent); }}
.lede {{ max-width:44rem; font-size:1.12rem; color:var(--ink2); margin:0 0 34px; }}
.paper-card {{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:26px 28px; margin-bottom:22px; }}
.paper-card h2 {{ font-family:var(--display); font-weight:400; font-size:1.65rem; line-height:1.18; margin:0 0 8px; }}
.paper-card h2 a {{ color:var(--ink); text-decoration:none; }} .paper-card h2 a:hover {{ color:var(--accent); }}
.byline {{ font-size:0.86rem; color:var(--muted); margin:0 0 14px; }}
.byline span {{ font-family:var(--mono); font-size:0.78rem; }}
.hi {{ margin:0 0 18px 1.2rem; padding:0; color:var(--ink2); font-size:0.94rem; }}
.hi li {{ margin-bottom:6px; }}
.actions {{ display:flex; flex-wrap:wrap; gap:10px; }}
.btn {{ display:inline-flex; align-items:center; padding:8px 15px; border-radius:999px; border:1px solid var(--ink); background:var(--ink); color:var(--bg); font-size:0.86rem; font-weight:500; text-decoration:none; cursor:pointer; font-family:var(--body); }}
.btn:hover {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
.btn.ghost {{ background:transparent; color:var(--ink); border-color:var(--line); }}
.btn.ghost:hover {{ border-color:var(--ink); background:transparent; color:var(--ink); }}
footer {{ margin-top:40px; padding-top:18px; border-top:1px solid var(--line); font-size:0.86rem; color:var(--muted); }}
</style>
</head>
<body>
<header class="top">
  <div class="bar">
    <a class="mark" href="/"><i aria-hidden="true"><b></b></i>Threebits</a>
    <nav>
      <a href="/">Overview</a>
      <a href="/#human">Human brain</a>
      <a href="/#results">Decoders</a>
      <a href="/gym/">Neural Gym</a>
      <a href="/#funding">Funding</a>
    </nav>
  </div>
</header>
<main class="wrap">
  <div class="kicker">Open Research Preprints</div>
  <h1>Threebits <em>research papers</em>.</h1>
  <p class="lede">Working preprints with full methods, negative controls, null replications, publication figures, print-ready PDFs, and arXiv LaTeX source bundles — built entirely on open public human and primate datasets.</p>
  {''.join(cards_html)}
  <footer>All human and primate datasets analyzed in these preprints are public open-access recordings (OpenNeuro CC0 &amp; DANDI CC-BY-4.0). Open reproducible code: <a href="https://github.com/MuhammadFarhantahir/threebits-research" target="_blank" rel="noopener">GitHub</a>. Last updated 21 November 2026.</footer>
</main>
<script>
document.querySelectorAll(".copy-btn").forEach(btn => {{
  btn.onclick = async () => {{
    try {{
      await navigator.clipboard.writeText(btn.dataset.bib);
      btn.textContent = "Copied BibTeX!";
      setTimeout(() => btn.textContent = "Copy BibTeX", 1800);
    }} catch(_) {{}}
  }};
}});
</script>
</body>
</html>
"""
    (WEB_PAPERS / "index.html").write_text(index_html)
    print("Built website/papers/index.html + 3 readable HTML papers + 3 PDFs + 3 arXiv bundles.")


if __name__ == "__main__":
    build_all()
