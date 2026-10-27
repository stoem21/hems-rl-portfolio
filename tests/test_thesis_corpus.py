"""Pure unit tests for rag_agent's LaTeX chunking/citation logic — no mocking
needed, no live Ollama needed. These don't touch _embed/_get_index at all.
"""
from rag_agent import _clean_and_chunk_tex, _parse_bib, _resolve_cites, _strip_latex


def test_table_block_kept_as_single_chunk(tmp_path):
    tex = tmp_path / "x.tex"
    tex.write_text(
        r"""
        Some intro text.

        \begin{table}[H]
        \caption{A caption}
        \label{tab:x}
        1 & 2 \\
        \end{table}

        More text after the table.
        """
    )
    chunks = _clean_and_chunk_tex(str(tex), bib_map={})
    table_chunks = [c for c in chunks if c["kind"] == "table"]
    assert len(table_chunks) == 1
    assert "A caption" in table_chunks[0]["text"]
    assert "\\label" not in table_chunks[0]["text"]


def test_prose_survives_around_table(tmp_path):
    tex = tmp_path / "x.tex"
    tex.write_text(
        r"""
        This is a long enough introductory paragraph to survive the minimum length filter.

        \begin{table}[H]
        \caption{A caption}
        \end{table}

        This is a long enough trailing paragraph to also survive the minimum length filter.
        """
    )
    chunks = _clean_and_chunk_tex(str(tex), bib_map={})
    prose_chunks = [c for c in chunks if c["kind"] == "prose"]
    assert any("introductory paragraph" in c["text"] for c in prose_chunks)
    assert any("trailing paragraph" in c["text"] for c in prose_chunks)


def test_cite_resolved_to_readable_form():
    bib_map = {"kell2021": "Kell et al., 2021"}
    text = r"Some prior work \cite{kell2021} showed something interesting."
    resolved = _resolve_cites(text, bib_map)
    assert "Kell et al., 2021" in resolved
    assert "\\cite" not in resolved


def test_cite_multiple_keys_resolved():
    bib_map = {"a2020": "A, 2020", "b2021": "B, 2021"}
    resolved = _resolve_cites(r"\cite{a2020,b2021}", bib_map)
    assert "A, 2020" in resolved
    assert "B, 2021" in resolved


def test_comment_lines_are_dropped(tmp_path):
    tex = tmp_path / "x.tex"
    tex.write_text(
        "% This whole line is a leftover template comment about something unrelated.\n"
        "\\chapter{Introduction}\n\n"
        "This is the real content paragraph that should survive chunking intact.\n"
    )
    chunks = _clean_and_chunk_tex(str(tex), bib_map={})
    joined = " ".join(c["text"] for c in chunks)
    assert "leftover template comment" not in joined
    assert "real content paragraph" in joined


def test_strip_latex_unwraps_textbf_and_removes_label():
    text = r"This is \textbf{bold} text with a \label{sec:x} label and a \ref{sec:y} reference."
    stripped = _strip_latex(text)
    assert "bold" in stripped and "\\textbf" not in stripped
    assert "\\label" not in stripped
    assert "[reference omitted]" in stripped


def test_parse_bib_extracts_author_and_year(tmp_path):
    bib = tmp_path / "x.bib"
    bib.write_text(
        "@techreport{irena2024,\n"
        "  author = {{International Renewable Energy Agency (IRENA)}},\n"
        "  title = {Some Report},\n"
        "  year = {2024},\n"
        "}\n"
    )
    bib_map = _parse_bib(str(bib))
    assert "irena2024" in bib_map
    assert "2024" in bib_map["irena2024"]
