# Decoding appendix

`7-decoding-appendix.tex` is the section added at the end of the Overleaf paper
on 2026-11-14. It contains Direct-Channel Classification and Side-Channel
Fidelity, with five result tables. The original body was unchanged.

The main document includes it after the bibliography:

```tex
\clearpage
\appendices
\input{tex-sections/7-decoding-appendix}
```

Copy this file into `tex-sections/` when integrating it. It uses the existing
IEEEtran document, float package, and body labels; it is not a standalone
document. The Overleaf build has nine pages, with the appendix on pages 8--9,
and zero compile errors.

`va-decode.tex`, `vb-phase.tex`, and `decode-bridge.tex` are complete numerical
table exports from `code/report.py`. They include more conditions than the
concise appendix. `vb-phase.tex` is a long tabular export for later selection
or splitting, not a single-column publication layout.

For interpretation and measurement limits, see the parent README and protocols.
