# AdvShield-Med, Review Sprint Plan

**Strategy:** We are ahead in implementation. Present only what each review evaluates, don't reveal all cards early.

---

## Project: Two-Part Framing

| Part | Name | What It Does | ML Phases |
|------|------|-------------|-----------|
| **A** | **Adversarial Detection** | Gatekeeper CNN detects tampered X-ray inputs before they reach the diagnostic model | Phase 1 (Baseline), Phase 2 (Attacks), Phase 3 (Gatekeeper), Phase 4 (Pipeline) |
| **B** | **Adversarial Purification & Recovery** | Purifier autoencoder repairs attacked images; backup model provides robust diagnosis | Phase 5a (Purifier), Phase 5b (Backup), Phase 5c (DeepFool), Phase 5d (Grad-CAM), Phase 6 (Sanitize-First) |

---

## Review-wise Sprint Mapping

### Review 1 ✅ (Guide Review, 9-11 July 2026), 5 Marks
**What was evaluated:** Title, abstract, problem, objectives, scope, preliminary plan
**Status:** COMPLETED

---

### Review 2 📌 (Panel Review, 19 August 2026), 20 Marks
**What is evaluated:** Domain understanding, literature review, methodology, system design

**What to present:**
- [x] **Domain:** Medical AI vulnerability to adversarial attacks in radiograph CAD systems
- [x] **Literature Review:** Minimum 15 recent papers (2020-2026) covering adversarial attacks, frequency/intermediate detection, autoencoder purification, and multi-layer medical AI defenses
- [x] **Methodology:** Two-part framework (Part A: Detection + Part B: Purification), conceptual level
- [x] **System Architecture:** Module diagram, data flow, dual operational modes (Strict Rejection vs Sanitize-First), tech stack
- [x] **Objectives, scope, expected outcomes**
- [x] **Feasibility, risk register, ethics, timeline**
- [x] **Report Chapters 1-3** (drafted in academic, humanized prose)

**What NOT to present yet:**
- ❌ Actual training results or accuracy numbers
- ❌ Working code demonstration
- ❌ Full-stack web app
- ❌ Grad-CAM visualizations

**Deliverables (Created in `Reviews/R2/`):**
- Report Chapter 1: `chapter1_introduction.md`
- Report Chapter 2: `chapter2_literature_review.md` (15 recent papers 2020-2026)
- Report Chapter 3: `chapter3_methodology.md`
- Canva AI PPT Prompt & 15-Slide Script: `canva_ppt_prompt.md`

---

### Review 3 (Panel Review, 16 September 2026), 20 Marks
**What is evaluated:** Follow-up on R2 feedback, ~50% implementation, interim results

**What to present:**
- [ ] Part A complete: Diagnostic model trained (96.13%), vulnerability demonstrated (PGD→0%), Gatekeeper trained (F1=0.996)
- [ ] Part A pipeline integrated: 100% clean pass, 99.2% PGD rejection
- [ ] Phase 1-4 results with tables, plots, training curves
- [ ] Address any R2 feedback
- [ ] Updated report: Chapters 1-3 revised + Chapter 4 started (implementation + interim results)

**What NOT to present yet:**
- ❌ Purifier / recovery mechanism (Part B)
- ❌ Backup model
- ❌ DeepFool evaluation
- ❌ Full-stack web app
- ❌ Grad-CAM

---

### Review 4 (Guide Review, 12-16 October 2026), 25 Marks
**What is evaluated:** Complete implementation, final results, report verification, final-review readiness

**What to present:**
- [ ] Part B complete: Purifier trained, backup model trained, DeepFool generalization tested
- [ ] Sanitize-first pipeline (Phase 6), final architecture with all attack types
- [ ] Grad-CAM interpretability visualizations
- [ ] Full-stack web app demo (FastAPI + React dashboard)
- [ ] Complete results across all 3 attack types × all defense mechanisms
- [ ] Near-final report: All 5 chapters complete
- [ ] Address R3 feedback

---

### Review 5 (Panel Review, 21 October 2026), 25 Marks
**What is evaluated:** Final technical evaluation, demonstration, report, presentation, viva

**What to present:**
- [ ] Live demo: Upload X-ray → Gatekeeper flag → Purifier repair → Diagnosis
- [ ] Complete results tables and analysis
- [ ] Final polished report (all chapters + appendices)
- [ ] Final presentation with all visualizations
- [ ] Individual contribution statement
- [ ] Publication/patent evidence (if applicable)

---

## Timeline Summary

```
July 9-11 ✅ R1: Title, abstract, problem definition
 ↓ (already implemented: Phases 1-6, full ML pipeline)
Aug 19 📌 R2: Domain, literature (15 papers 2020-2026), methodology, design (NO results)
 ↓ (present Part A results at R3)
Sep 16 R3: Part A implementation + interim results (~50%)
 ↓ (present Part B + full-stack at R4)
Oct 12-16 R4: Part B + full app + complete report
 ↓ (polish everything)
Oct 21 R5: Final demo + polished report + viva
```
