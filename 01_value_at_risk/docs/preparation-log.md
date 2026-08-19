# Preparation log — what was closed before submission

> Moved out of `paper/draft-v1.md`. This is project management, not part of the manuscript.
> Retained because it records which conditions were raised by which reviewer role and how
> each was discharged — the audit trail behind the paper's claim to have been reviewed
> adversarially. Every line was `[x]` at the time of the move.

## Closed items

- [x] Bootstrap intervals for every ratio quoted (Risk F7) — `outputs/bootstrap_intervals.json`,
      now derived and drift-checked alongside the figures file. Its two synthetic entries had
      also been left at the superseded four-weight grid; see **Appendix B.2**.
- [x] Figure 1: dose–response, weight vs IQR ratio, ρ = +0.585 (`scripts/make_figures.py`).
- [x] Figure 2: (a) real vs permuted prior, six paired cells; (b) synthetic grid with the
      analytical contraction overlaid.
- [x] **Section H — the control's lineage.** Closed: Fisher (1935), Pitman (1937), Ojala &
      Garriga (2010), Zhang et al. (2017), Adebayo et al. (2018), Bertrand et al. (2004),
      Summers & Dinneen (2021) verified. §3.4, §3.5 and contribution 3 rewritten to credit them;
      the novelty claim is now a search null, not a gap.
- [x] **Section F — the ML-VaR survey. E4 MET.** Five ML-VaR papers read in full and coded
      (P1–P4, P6), plus P5 for the §3.4 premise. The survey cost the draft two framings — §1 and
      §3.4 — both revised downward. Open lead: Madani et al. (2004), unread.
- [x] **H.6 — closed, and it falsified the framing.** Bhojanapalli et al. (2021) read in full:
      they do propose regularisation for run-to-run variance, and they handle it correctly.
      §3.4/§3.5 narrowed accordingly — the error is inferring from stability a conclusion about
      the target, not reporting stability. Petneházi (2019) is now the one documented instance
      of the inference being made.
- [x] Related work, pre-registration strand: **closed** — Arpinon & Espinosa (2023) for the
      economics guide; Lin et al. (2024) for adoption rates. **≤ 1% of journals in economics
      had adopted registered reports as of July 2023**, so §1 can now quantify the rarity of
      this paper's own protocol instead of asserting it.
- [x] **Related work complete.** Final strand — selection under small validation blocks — closed
      as **Section I** of `references.md`: Cawley & Talbot (2010), Varma & Simon (2006),
      Bengio & Grandvalet (2004), Bergstra & Bengio (2012). §3.2 rewritten to cite the mechanism
      rather than only the power calculation, and §8 gained a limitation (single chronological
      partition) that this reading exposed.
- [x] **Citation verification pass complete. 47 verified, 0 from memory, 0 placeholders.**
      **Five of twenty were wrong** — Fisher's chapter, Gelman & Loken's venue and year,
      Madani et al.'s authors *and* its substance, Taylor (2019)'s method, Lin et al.'s year.
      Four of the five were inherited from other people's descriptions of a source. Written up
      as **§5.1**. One claim (*Pacific-Basin Finance Journal* replication platform) could not be
      sourced and was **dropped rather than softened**.
- [x] Venue framing decided: **methods paper**, VaR as vehicle (§1 rewritten accordingly).
