# pytq: Open-Source Release Readiness

## Overview

Make the pytq library ready for public GitHub release with proper documentation, licensing, attribution, and usability polish.

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `README.md` | Create | Primary documentation with install, usage, benchmarks, attribution |
| `LICENSE` | Create | Apache 2.0 license |
| `CITATION.cff` | Create | Machine-readable citation for the original TurboQuant paper |
| `.gitignore` | Update | Cover all generated artifacts |
| `pyproject.toml` | Update | Add project URLs, classifiers, author info |
| `index.html` | Update | Add GitHub repo link |

## README.md Structure

1. **Header**: `pytq` name + one-line description + badges (Python >=3.9, Apache 2.0, tests passing)
2. **Attribution block** (immediately after header): "An independent PyTorch implementation of TurboQuant by Zandieh, Daliri, Hadian & Mirrokni (ICLR 2026). This project is not affiliated with or endorsed by the original authors or Google Research."
3. **Key results**: 3-column highlight (1.75x compression, 0.4ms latency, 0% quality loss)
4. **Installation**: `pip install pytq` + from-source instructions
5. **Quick Start**: 3 code examples (TurboQuantMSE, TurboQuantProd, TurboQuantKVCache)
6. **API Reference**: Brief docs for each public class with constructor args and methods
7. **Benchmarks**: How to run, what they measure, summary table of results
8. **How It Works**: 4-step algorithm summary (normalize, rotate, quantize, dequantize) with theoretical guarantee
9. **Citation**: BibTeX entry for the original paper (not this repo)
10. **License**: Apache 2.0 note

## LICENSE

Apache License 2.0 — standard text with current year and author name.

## CITATION.cff

Points to the **original paper** by Zandieh et al., not this repository. Includes:
- Paper title, authors, DOI/arXiv ID
- Conference: ICLR 2026
- Note that this is a citation for the algorithm, not the implementation

## pyproject.toml Updates

- Add `[project.urls]` with Homepage, Repository, Paper links
- Add `license` field
- Add `classifiers` for PyPI (Development Status, License, Python versions, Topic)
- Add `authors` field

## .gitignore Updates

Ensure coverage of:
- `.venv/`
- `results/`
- `benchmark_results/`
- `*.egg-info/`
- `__pycache__/`
- `.pytest_cache/`

## index.html Update

Add a GitHub link button in the hero section pointing to the repository.

## Usability Checks

- Verify `pip install -e .` works from fresh clone
- Verify `from pytq import TurboQuantMSE` works
- Verify all 37 tests pass
- Check error messages for invalid inputs (wrong dim, bits < 1, etc.)

## Out of Scope

- No separate docs site
- No CI/CD setup
- No PyPI publishing
- No API changes to existing code
