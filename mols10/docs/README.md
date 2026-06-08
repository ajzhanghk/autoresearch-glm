# MOLS(10) Research Documentation

This directory contains comprehensive documentation of a computational search for three
mutually orthogonal Latin squares (MOLS) of order 10 — the **N(10)** problem.

## Document Index

| # | Document | Contents |
|---|----------|---------|
| 01 | [Problem Statement](01_problem_statement.md) | Background, history, equivalences, research goal |
| 02 | [Mathematical Theory](02_mathematical_theory.md) | Common transversals, energy function, SAT encoding, CT barrier theorem |
| 03 | [Algorithms](03_algorithms.md) | All 5 search algorithms: sa_triple_pt, multi_decomp, sa_ct_climb, reverse search, SAT worker |
| 04 | [Experimental Results](04_experimental_results.md) | Energy cascade, CT distribution, SAT results, reverse search results, local minimum analysis |
| 05 | [Near-Miss Analysis](05_near_miss_analysis.md) | Deep analysis of E=37 triples: structure, diversity, pool management, statistical interpretation |
| 06 | [CT Barrier Analysis](06_ct_barrier.md) | Why CT≤2 for all known pairs; evidence, algebraic motivation, SAT certificates |
| 07 | [Implementation Details](07_implementation.md) | Codebase, data structures, key algorithms, bugs table, reproducibility |
| 08 | [Conclusions & Future Work](08_conclusions_and_future_work.md) | Main conclusion, open questions, conjectures, future directions |
| 09 | [Appendix: Data Tables](09_appendix_data.md) | Raw data, histograms, session timeline, notation reference |

## Quick Reference

### Key Result

> After Sessions 1–7 with 7 parallel workers:
> - **Best energy found:** $E = 37$ ($\mathrm{cl}_{12}=0$, $\mathrm{cl}_{13}=22$, $\mathrm{cl}_{23}=15$)
> - **CT maximum found:** 2 (across 311 MOLS-10 pairs)
> - **SAT verdict:** All 311+ pairs proven UNSAT by Glucose3
> - **$N(10) \geq 3$ proven:** **No** (search ongoing)

### Evidence Summary for N(10) = 2

1. CT ≤ 2 for all 311 tested MOLS-10 pairs (CT < 10 → no L3)
2. SAT UNSAT certificate for all 311 pairs (definitive per-pair proofs)
3. E=37 energy barrier: strict 2-step local minimum, never escaped
4. Reverse search: near-miss L3s incompatible with any MOLS pair (E_rev ≥ 60)

## Paper Outline

For an academic paper based on this work, suggested sections:

1. **Introduction** — N(10) problem, prior work, computational approach
2. **Background** — Latin squares, orthogonality, common transversals, energy formulation
3. **The CT Barrier** — Theorem, enumeration results, SAT certification
4. **Search Methodology** — Parallel-tempering SA, ILS, initialization portfolio
5. **Near-Miss Analysis** — E=37 triples, local minimum structure, reverse search
6. **Experimental Evaluation** — Statistics, session history, hardware
7. **Discussion** — Implications for N(10), conjectures
8. **Conclusion** — Summary, future work

## Repository

- **Branch:** `claude/mols-order-10-search-yfQXK`
- **Repo:** `ajzhanghk/autoresearch-glm`
- **Search code:** `mols10/` directory
- **Results:** `mols10/results/`
