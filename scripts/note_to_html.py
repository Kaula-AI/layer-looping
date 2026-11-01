"""Render docs/findings.md as the kaula.ai note page (the website repo's dist/notes/<slug>/index.html).

Handles the Markdown this note uses: headings, paragraphs, bullet and numbered lists, bold, italics,
inline code, links, images and fenced code. Figures are copied next to the page.

  python scripts/note_to_html.py --site ../website --slug layer-looping
"""
import argparse
import html
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "https://github.com/Kaula-AI/layer-looping"
MARK = ('<svg class="brand-symbol" viewBox="0 0 64 64" aria-hidden="true"><path fill="currentColor" d="M32 5l8 '
        '14h-16ZM22 23l8 14h-16ZM42 23l8 14h-16ZM12 41l8 14h-16ZM32 41l8 14h-16ZM52 41l8 14h-16Z"/></svg>')


def inline(t, figs):
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\w)", r"<em>\1</em>", t)

    def link(m):
        text, url = m.group(1), m.group(2)
        if not re.match(r"https?://", url):  # repo-relative link -> GitHub
            url = f"{REPO}/blob/main/{os.path.normpath(os.path.join('docs', url))}"
        return f'<a href="{url}">{text}</a>'

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, t)


def render(md, figs):
    lines = md.splitlines()
    out, i, title = [], 0, ""
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            while not lines[j].startswith("```"):
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i + 1: j])) + "</code></pre>")
            i = j + 1
            continue
        m = re.match(r"(#{1,3}) (.*)", ln)
        if m:
            lvl, text = len(m.group(1)), m.group(2)
            if lvl == 1:
                title = text
            else:
                slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
                out.append(f'<h{lvl} id="{slug}">{inline(text, figs)}</h{lvl}>')
            i += 1
            continue
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", ln)
        if m:
            alt, src = m.group(1), m.group(2)
            figs.append(src)
            cap = ""
            if i + 2 < len(lines) and lines[i + 1] == "" and re.match(r"\*.*\*$", lines[i + 2]):
                cap = f"<figcaption>{inline(lines[i + 2][1:-1], figs)}</figcaption>"
                i += 2
            out.append(f'<figure><img src="{os.path.basename(src)}" alt="{html.escape(alt)}" loading="lazy">{cap}</figure>')
            i += 1
            continue
        if re.match(r"(- |\d+\. )", ln):
            tag = "ol" if ln[0].isdigit() else "ul"
            items = []
            while i < len(lines) and re.match(r"(- |\d+\. )", lines[i]):
                items.append(re.sub(r"^(- |\d+\. )", "", lines[i]))
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x, figs)}</li>" for x in items) + f"</{tag}>")
            continue
        if ln.strip():
            para = [ln]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"(#|- |\d+\. |!\[|```)", lines[i]):
                para.append(lines[i])
                i += 1
            text = " ".join(para)
            if re.match(r"\*.*\*$", text):  # standalone italic line: the byline
                out.append(f'<p class="note-byline">{inline(text[1:-1], figs)}</p>')
            else:
                out.append(f"<p>{inline(text, figs)}</p>")
            continue
        i += 1
    return title, "\n".join(out)


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#eae8e1">
  <meta name="description" content="{desc}">
  <meta property="og:type" content="article">
  <meta property="og:title" content="{title_attr}">
  <meta property="og:description" content="{desc}">
  <meta property="og:url" content="https://kaula.ai/notes/{slug}/">
  <meta property="og:image" content="https://kaula.ai/notes/{slug}/{og_image}">
  <meta name="twitter:card" content="summary_large_image">
  <link rel="canonical" href="https://kaula.ai/notes/{slug}/">
  <link rel="icon" href="/assets/favicon.svg?v=3" type="image/svg+xml">
  <link rel="stylesheet" href="/styles.css">
  <link rel="stylesheet" href="/notes/notes.css">
  <title>{title_text} — Kāula</title>
</head>
<body>
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="site-header note-header">
    <a class="brand" href="/" aria-label="Kāula home">{mark}<span>kāula</span></a>
    <nav class="note-nav" aria-label="Main navigation"><a href="/#research">Research</a><a href="/notes/">Notes</a><a href="{repo}">Code</a></nav>
  </header>
  <main id="main" class="note">
    <p class="eyebrow note-eyebrow">Technical note</p>
    <h1>{title_html}</h1>
    {body}
    <p class="note-end"><a href="/notes/">All notes</a> · <a href="{repo}">Code and data</a></p>
  </main>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True, help="website repo root")
    p.add_argument("--slug", default="layer-looping")
    p.add_argument("--src", default=os.path.join(ROOT, "docs", "findings.md"))
    a = p.parse_args()
    md = open(a.src).read()
    figs = []
    title, body = render(md, figs)
    first = re.search(r"<p>(.*?)</p>", body).group(1)
    desc = re.sub(r"<[^>]+>", "", first)[:240]
    out_dir = os.path.join(a.site, "dist", "notes", a.slug)
    os.makedirs(out_dir, exist_ok=True)
    for f in figs:
        shutil.copy(os.path.join(os.path.dirname(a.src), f), os.path.join(out_dir, os.path.basename(f)))
    page = PAGE.format(desc=html.escape(desc), title_attr=html.escape(title), title_text=html.escape(title),
                       title_html=inline(title, []), slug=a.slug, og_image=os.path.basename(figs[-1]),
                       mark=MARK, repo=REPO, body=body)
    open(os.path.join(out_dir, "index.html"), "w").write(page)
    print("wrote", out_dir, "with", len(figs), "figures")


if __name__ == "__main__":
    main()
