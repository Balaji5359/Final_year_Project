# Reviewer Report (Prototype Paper)

## Summary Score
- Overall score: **7.1 / 10**
- Recommendation: **Weak Accept**

## Criterion-wise Assessment
- Problem relevance (High): **8.5/10**
- Technical correctness (High): **7.5/10**
- Experimental design (Medium): **6.5/10**
- Writing quality (Medium): **7.5/10**
- Novelty (Medium): **6.0/10**

## Strengths
1. Addresses a practical and relevant airport operations problem.
2. Integrates prediction and optimization in one coherent pipeline.
3. Uses clear constraints and reports objective trade-offs transparently.
4. Provides reproducible outputs through CSV artifacts and dashboard.

## Weaknesses
1. Evaluation is limited to synthetic data and one scenario family.
2. No benchmark against exact solvers or stronger baselines.
3. Environmental outcome is based on a proxy rather than validated emissions model.
4. Novelty is moderate relative to existing GA-based AGAP literature.

## Suggestions for Improvement
1. Add multiple random-seed runs and confidence intervals for robustness.
2. Compare against MILP or additional heuristics under identical constraints.
3. Include ablation of objective weights to quantify trade-off sensitivity.
4. Expand discussion on computational complexity and runtime behavior.

## Final Decision Rationale
As a student prototype, the study is methodologically sound and clearly presented, with meaningful but bounded contributions. The current evidence is sufficient for a weak accept in a student or prototype-oriented venue, but not yet strong for a top-tier systems conference.
