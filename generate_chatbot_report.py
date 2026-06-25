"""
Génération du rapport PDF — Agent IA SmartRecovery (Chatbot Q&A uniquement).
10 questions métier posées au chatbot Gemini + LangChain, dont 2 en Darija.
Ce rapport est INDÉPENDANT du rapport modèles ML.
"""

import json
import re
from datetime import date
from pathlib import Path
from weasyprint import HTML

BASE_DIR = Path(__file__).resolve().parent

# ── Chargement des Q&A ────────────────────────────────────────────────────────
QA_FILE = BASE_DIR / "agent_qa.json"
if not QA_FILE.exists():
    raise FileNotFoundError(f"Fichier introuvable : {QA_FILE}")

with open(QA_FILE, encoding="utf-8") as f:
    qa_items = json.load(f)

print(f"Chargement de {len(qa_items)} Q&A depuis agent_qa.json...")

# ── Helpers HTML ──────────────────────────────────────────────────────────────

def md_to_html(text: str) -> str:
    """Convertit le markdown minimal des réponses en HTML."""
    # Tableaux markdown → HTML table
    lines = text.split("\n")
    result_lines = []
    in_table = False
    table_buffer = []

    for line in lines:
        if re.match(r"^\s*\|.+\|", line):
            if not in_table:
                in_table = True
                table_buffer = []
            table_buffer.append(line)
        else:
            if in_table:
                result_lines.append(_render_md_table(table_buffer))
                table_buffer = []
                in_table = False
            result_lines.append(line)
    if in_table and table_buffer:
        result_lines.append(_render_md_table(table_buffer))

    text = "\n".join(result_lines)
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Paragraphes
    text = re.sub(r"\n\n", "</p><p>", text)
    # Listes numérotées et à puces
    text = re.sub(r"\n([0-9]+\. )", r"<br/>\1", text)
    text = re.sub(r"\n([-•✅⚠️⚡🔑🎯💡📋📊🏆📈🔍💰📅🏛️🔨] )", r"<br/>\1", text)
    text = re.sub(r"\n", "<br/>", text)
    return f"<p>{text}</p>"


def _render_md_table(lines: list) -> str:
    """Convertit un bloc de lignes markdown tableau en <table> HTML."""
    html = '<table class="md-table">'
    is_header = True
    for line in lines:
        # Ignorer les lignes séparateurs |---|---|
        if re.match(r"^\s*\|[\s\-:]+\|", line):
            is_header = False
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        tag = "th" if is_header else "td"
        row_class = ' class="header-row"' if is_header else ""
        html += f"<tr{row_class}>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
        if is_header:
            is_header = False
    html += "</table>"
    return html


def tools_badges(tools: list) -> str:
    colors = {
        "predict_dossier":   "#3b82f6",
        "query_history":     "#10b981",
        "get_segment_scoring": "#8b5cf6",
    }
    labels = {
        "predict_dossier":   "predict_dossier",
        "query_history":     "query_history",
        "get_segment_scoring": "get_segment_scoring",
    }
    out = ""
    for t in tools:
        c = colors.get(t, "#6b7280")
        l = labels.get(t, t)
        out += (
            f'<span style="background:{c};color:white;border-radius:3px;'
            f'padding:2px 8px;font-size:7.5pt;margin-right:4px;'
            f'font-weight:600;letter-spacing:0.2px;">{l}</span>'
        )
    return out


def darija_badge() -> str:
    return (
        '<span style="background:#b45309;color:white;border-radius:3px;'
        'padding:2px 9px;font-size:7.5pt;font-weight:700;margin-left:8px;">'
        'Darija (MA)</span>'
    )


# ── Génération des cartes Q&A ─────────────────────────────────────────────────
cards_html = ""
for i, qa in enumerate(qa_items, 1):
    lang        = qa.get("language", "fr")
    is_darija   = (lang == "darija")
    tools_html  = tools_badges(qa.get("tools_used", []))
    raw_answer  = qa["answer"]
    is_api_err  = raw_answer.strip().startswith("Erreur de l'agent")
    if is_api_err:
        # Afficher un bandeau d'attente propre au lieu du stacktrace brut
        answer_html = (
            '<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:4px;'
            'padding:12px 16px;color:#991b1b;font-size:8.5pt;">'
            '<strong>Réponse en attente</strong> — La clé API Gemini a atteint son quota journalier '
            '(free tier). Le script <code>refresh_qa_and_generate_pdf.py</code> renverra '
            'automatiquement cette question au prochain reset du quota (minuit PT).'
            '</div>'
        )
    else:
        answer_html = md_to_html(raw_answer)
    lang_badge  = darija_badge() if is_darija else ""

    # Couleur de l'en-tête selon langue
    header_bg   = "#1e3a5f" if is_darija else "#0f172a"
    q_bg        = "#f8fafc" if is_darija else "#f8fafc"
    q_border    = "#cbd5e1" if is_darija else "#e2e8f0"
    q_color     = "#1e3a5f" if is_darija else "#0f172a"

    cards_html += f"""
    <div class="qa-card" style="page-break-inside:avoid;">
      <div class="qa-header" style="background:{header_bg};">
        <span class="qa-num">Question {i} / {len(qa_items)}</span>
        {lang_badge}
        <span class="qa-key">{qa['key']}</span>
      </div>
      <div class="qa-question" style="background:{q_bg};border-bottom:1px solid {q_border};color:{q_color};">
        <em>« {qa['question']} »</em>
      </div>
      <div class="qa-tools">
        <span class="tools-label">Outils invoqués :</span> {tools_html}
      </div>
      <div class="qa-answer">
        <span class="agent-label">Agent SmartRecovery :</span>
        {answer_html}
      </div>
    </div>"""

# ── Statistiques résumées ─────────────────────────────────────────────────────
nb_darija  = sum(1 for q in qa_items if q.get("language") == "darija")
nb_fr      = len(qa_items) - nb_darija
nb_ok      = sum(1 for q in qa_items if not q.get("answer","").startswith("Erreur de l'agent"))
nb_pending = len(qa_items) - nb_ok
tools_count: dict = {}
for q in qa_items:
    for t in q.get("tools_used", []):
        tools_count[t] = tools_count.get(t, 0) + 1

stats_rows = [
    ("Questions totales", len(qa_items)),
    ("En français", nb_fr),
    ("En Darija (MA)", nb_darija),
    ("Réponses OK", nb_ok),
]
if nb_pending:
    stats_rows.append(("En attente", nb_pending))

stats_html = "".join(
    f'<div class="stat-box"><div class="stat-val">{v}</div>'
    f'<div class="stat-lbl">{k}</div></div>'
    for k, v in stats_rows
)

# ── HTML complet ──────────────────────────────────────────────────────────────
html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<style>
  @page {{ size: A4; margin: 14mm 16mm; }}
  body {{
    font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
    color: #1e293b; line-height: 1.55; margin: 0; padding: 0;
    background: #f8fafc;
  }}

  /* ── En-tête ── */
  .report-header {{
    background: #0f172a;
    color: white; padding: 26px 32px 20px; text-align: center;
    margin-bottom: 18px;
    border-bottom: 4px solid #2563eb;
  }}
  .report-header h1 {{ margin: 0 0 6px; font-size: 17pt; letter-spacing: 0.3px; font-weight: 700; }}
  .report-header .sub {{
    font-size: 9.5pt; opacity: 0.85; margin: 0 0 4px;
  }}
  .report-header .meta {{
    font-size: 8pt; opacity: 0.6; margin: 0;
  }}

  /* ── Badge architecture ── */
  .arch-banner {{
    background: #f1f5f9; border-left: 4px solid #2563eb;
    border-radius: 5px; padding: 10px 16px; margin-bottom: 16px;
    font-size: 8.5pt; color: #334155; line-height: 1.6;
  }}

  /* ── Stats bar ── */
  .stats-bar {{
    display: flex; gap: 8px; margin-bottom: 18px; flex-wrap: wrap;
  }}
  .stat-box {{
    flex: 1; background: white; border-radius: 5px;
    padding: 10px 8px; text-align: center;
    border-top: 3px solid #1e3a8a;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    min-width: 80px;
  }}
  .stat-val {{ font-size: 16pt; font-weight: 800; color: #0f172a; }}
  .stat-lbl {{ font-size: 7pt; color: #64748b; text-transform: uppercase; font-weight: 600; margin-top: 2px; letter-spacing: 0.3px; }}

  /* ── Carte Q&A ── */
  .qa-card {{
    background: white; border-radius: 5px; margin-bottom: 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07); overflow: hidden;
    border: 1px solid #e2e8f0;
  }}
  .qa-header {{
    padding: 10px 16px; display: flex; align-items: center;
    gap: 8px;
  }}
  .qa-num {{
    color: white; font-size: 9pt; font-weight: 700;
    flex-shrink: 0;
  }}
  .qa-key {{
    margin-left: auto; color: rgba(255,255,255,0.55);
    font-size: 7.5pt; font-family: monospace;
  }}
  .qa-question {{
    padding: 11px 16px; font-size: 9.5pt;
    font-style: italic; line-height: 1.55;
  }}
  .qa-tools {{
    background: #f8fafc; padding: 5px 16px;
    border-top: 1px solid #f1f5f9;
    border-bottom: 1px solid #f1f5f9;
  }}
  .tools-label {{ font-size: 7.5pt; color: #94a3b8; }}
  .qa-answer {{
    padding: 12px 16px 6px; font-size: 9pt; color: #1e293b; line-height: 1.65;
  }}
  .agent-label {{ color: #1e3a8a; font-weight: 700; font-size: 9pt; }}
  .qa-answer p {{ margin: 4px 0; }}

  /* ── Tableaux markdown ── */
  .md-table {{
    width: 100%; border-collapse: collapse; margin: 8px 0;
    font-size: 8pt;
  }}
  .md-table th {{
    background: #1e3a5f; color: white;
    padding: 5px 8px; text-align: left;
    border: 1px solid #2563eb;
  }}
  .md-table td {{
    padding: 4px 8px; border: 1px solid #e2e8f0;
    vertical-align: top;
  }}
  .md-table tr:nth-child(even) td {{ background: #f8fafc; }}
  .md-table tr:nth-child(odd) td  {{ background: white; }}

  /* ── Pied de page ── */
  .report-footer {{
    text-align: center; font-size: 7.5pt; color: #94a3b8;
    margin-top: 16px; padding: 10px;
    border-top: 1px solid #e2e8f0;
  }}
</style>
</head>
<body>

<div class="report-header">
  <h1>Agent IA SmartRecovery — Rapport Q&A Chatbot</h1>
  <p class="sub">
    {len(qa_items)} Questions Métier · Powered by Google Gemini 2.0 Flash
    · LangChain ReAct Agent · 3 Tools PySpark ML
  </p>
  <p class="meta">
    Généré le {date.today().strftime("%d/%m/%Y")}
    &nbsp;·&nbsp; {nb_darija} question(s) en Darija testée(s)
    &nbsp;·&nbsp; Rapport indépendant des modèles ML
  </p>
</div>

<div class="arch-banner">
  <strong>Architecture IA :</strong>
  Question en langage naturel (français / Darija)
  → <strong>Gemini 2.0 Flash</strong> (raisonnement LLM)
  → <strong>LangChain ReAct</strong> (orchestration outils)
  → Tools PySpark ML
  (<em>predict_dossier · query_history · get_segment_scoring</em>)
  → Réponse structurée &amp; recommandation opérationnelle
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <strong>Mémoire :</strong> ConversationBufferMemory par session_id
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <strong>Max iterations :</strong> 5
</div>

<div class="stats-bar">
  {stats_html}
</div>

{cards_html}

<div class="report-footer">
  SmartRecovery Agent IA — LangChain + Gemini 2.0 Flash + PySpark ML
  · Rapport généré automatiquement depuis agent_qa.json
</div>

</body>
</html>"""

# ── Export HTML + PDF ─────────────────────────────────────────────────────────
output_html = str(BASE_DIR / "chatbot_qa_report.html")
output_pdf  = str(BASE_DIR / "chatbot_qa_report.pdf")

with open(output_html, "w", encoding="utf-8") as f:
    f.write(html_content)

print("Conversion HTML → PDF en cours...")
HTML(output_html).write_pdf(output_pdf)

print(f"✅ Rapport HTML : {output_html}")
print(f"✅ Rapport PDF  : {output_pdf}")
print(f"   {len(qa_items)} questions — {nb_darija} en Darija — {nb_fr} en français")
