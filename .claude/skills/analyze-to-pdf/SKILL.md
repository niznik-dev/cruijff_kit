---
name: analyze-to-pdf
description: Convert experiment analysis reports (report.md) to PDF using pandoc. Use after analyze-experiment to create shareable PDF reports.
---

# Analyze to PDF

Convert a markdown file to PDF using pandoc. Designed for experiment analysis reports but works with any markdown file.

## Your Task

1. Check that required system tools are installed
2. Locate the markdown file to convert
3. Convert to PDF with pandoc
4. Report the result

## Dependency Check

Verify tools are available before proceeding:

### Required: pandoc

```bash
which pandoc
```

If missing, stop and report:
```
pandoc is not installed. Install it with your package manager (e.g., apt install pandoc, dnf install pandoc).
```

### Required: PDF engine (one of the following, in priority order)

```bash
which xelatex    # preferred — better Unicode support
which pdflatex   # fallback
```

If neither is found, stop and report:
```
No PDF engine found. pandoc needs a LaTeX installation to produce PDFs.
Install texlive (e.g., apt install texlive-xetex or dnf install texlive-xetex).
```

Record which engine is available for the conversion step.

## Locate Markdown File

**If user provides a path argument:** Use that path directly.

**If no argument provided:**
1. Check if current directory contains `experiment_summary.yaml`
2. If yes, look for `analysis/report.md`
3. If that file exists, use it
4. Otherwise, ask the user for a path

**Validate** the file exists before proceeding.

## Preprocess for PDF

Reports may contain HTML `<details>` blocks (collapsible sections) that LaTeX cannot render. Expand them before conversion:

```python
from pathlib import Path
from tools.inspect.report_generator import expand_details_for_pdf

md_path = Path("{full_path_to_md}")
text = md_path.read_text()
expanded = expand_details_for_pdf(text)

# Write to a temp file in the same directory (so relative image paths still work)
tmp_path = md_path.with_suffix(".tmp.md")
tmp_path.write_text(expanded)
```

This converts `<summary>` text to bold labels and `<pre><code>` blocks to fenced code blocks. If the markdown has no `<details>` blocks, the text passes through unchanged.

## Convert to PDF

**Critical:** Change to the markdown file's parent directory before running pandoc. This ensures relative image paths (e.g., `![](scores_by_task.png)`) resolve correctly.

```bash
cd {parent_directory}
pandoc {tmp_filename} -o {stem}.pdf --pdf-engine={engine} \
  --from markdown-implicit_figures \
  -V geometry:margin=1in
```

**Why these flags matter:**

- `--from markdown-implicit_figures` — disables pandoc's automatic wrapping of images in LaTeX `\begin{figure}` float environments. Without this, LaTeX treats every image as a float and reorders them past surrounding text onto later pages. With this flag, images are inline `\includegraphics` calls that appear exactly where they are in the markdown, just like text. The tradeoff is no `Figure N:` captions, but report images already have `###` headings so captions are redundant.
- `geometry:margin=1in` — maximizes page real estate for figures

Where `{stem}` is the **original** filename without the `.md` extension (e.g., `report.md` → `report.pdf`).

After conversion, clean up the temp file:

```python
tmp_path.unlink()
```

## Report Result

**On success:**
```
PDF created: {full_path_to_pdf} ({file_size})
```

**On failure:** Show the pandoc error output. Common issues:
- Missing LaTeX packages: suggest `tlmgr install {package}` or a fuller texlive install
- Image format not supported: suggest converting images to PNG first
- Unicode issues with pdflatex: suggest installing xelatex instead

## Important Notes

- **No logging file** — this is a lightweight utility, not a multi-stage workflow
- **Idempotent** — re-running overwrites the existing PDF
- **Image embedding** — the `cd` into the file's directory is what makes relative image paths work. Do not skip this.
- **xelatex over pdflatex** — prefer xelatex when available for better Unicode and font handling
