# wormscope — reading list for the to-do list

Compiled 23 Aug 2026. Every link below was checked to resolve to the paper named.
Ranked most-useful-first inside each section. `[DOC]` = documentation/spec, everything else is a paper.

---

## If you only read six things

1. **Yemini et al. 2013, *Nature Methods* — A database of C. elegans behavioral phenotypes.** The canonical definitions of speed, length, area, curvature from a skeleton. Match these or document your deviation. https://pmc.ncbi.nlm.nih.gov/articles/PMC3962822/
2. **Archit et al. 2025, *Nature Methods* — Segment Anything for Microscopy (µSAM).** Answers your "how do I change segmentations after segment-all" question: it trains SAM by feeding the *previous iteration's predicted mask back in as a prompt*. https://doi.org/10.1038/s41592-024-02580-4
3. **Shappell et al. 2026, *PLOS Comp Biol* — PumpKin.** Detect → track → quantify pulsation in freely moving worms, with code. Your pulsation section, end to end. https://doi.org/10.1371/journal.pcbi.1014489
4. **Zulueta-Coarasa et al. 2025, *Nature Methods* — MIFA.** How to package and share an annotated bioimage dataset so it is actually reusable. https://doi.org/10.1038/s41592-025-02835-8
5. **Ouyang et al. 2023, *Front. Bioinformatics* — Moving beyond the desktop.** ~1 in 5 image.sc forum threads involve installation. The evidence for why deployment is real work. https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2023.1233748/full
6. **[DOC] micro-sam installation docs.** The closest comparable project (torch + SAM + Qt, aimed at biologists) *deprecated* its standalone installers and recommends conda. Read before you build an installer. https://computational-cell-analytics.github.io/micro-sam/micro_sam.html

---

## 1. Datastructure — video, frame, segmentation, label

**Start with the OME-Zarr spec and the VIS format; they are the two realistic targets.**

- **[DOC] OME-Zarr / NGFF spec v0.4** — https://ngff.openmicroscopy.org/0.4/
  §3.6 `labels` and §3.7 `image-label` model exactly your case: one integer per instance, plus a `colors`/`properties` table where `"worm"` / `"eggs"` live. Chunked, so a mostly-empty T×Y×X label array costs almost nothing.
- **Moore et al. 2021, *Nat Methods* — OME-NGFF** — https://doi.org/10.1038/s41592-021-01326-w
  The citable rationale for choosing Zarr over TIFF/HDF5. Two paragraphs to justify the format to a PI.
- **Yang, Fan & Xu 2019 — Video Instance Segmentation (YouTube-VIS)** — https://arxiv.org/abs/1905.04804
  Closest published match to your data model: COCO-style JSON where each annotation is a *list* of per-frame segmentations plus a video-level id and category.
- **Lin et al. 2014 — Microsoft COCO** — https://arxiv.org/abs/1405.0312
  The lingua franca export. `categories` = your labels, `annotations.segmentation` = your masks.
- **[DOC] pycocotools mask API (RLE)** — https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
  How to store thousands of per-frame masks without filling the disk: run-length encoding scales with boundary length, and `area`/`iou` work directly on the RLE (also how you compute worm-worm contact later).
- **Voigtlaender et al. 2019 — MOTS** — https://arxiv.org/abs/1902.03604
  Pixel-accurate masks *with persistent IDs across frames*. This is your problem the moment a worm keeps its identity between frames.
- **Javer et al. 2018, *Nat Methods* — Tierpsy + WCON** — https://doi.org/10.1038/s41592-018-0112-1
  The worm community's own JSON format (positions → full skeletons). If wormscope exports WCON, everything downstream in the field reads it.
- **Pont-Tuset et al. 2017 — DAVIS 2017** — https://arxiv.org/abs/1704.00675
  The flat indexed-PNG-per-frame alternative, plus the J&F metric for scoring annotator-vs-annotator agreement.
- **[DOC] CVAT export format table** — https://docs.cvat.ai/docs/dataset_management/formats/
  Which format loses the least information if you want interoperability instead of a silo.
- **[DOC] Ultralytics YOLO-seg dataset format** — https://docs.ultralytics.com/datasets/segment/
  Exact export needed if you ever fine-tune YOLO-seg. Polygon-only, lossy for holes.
- **Allan et al. 2012, *Nat Methods* — OMERO** — https://doi.org/10.1038/nmeth.1896
  The reference design for annotations as first-class, owned, server-side objects — relevant the moment there is more than one annotator.

**Provenance (who annotated / corrected what) — there is no bioimaging standard. Copy a tool schema:**

- **[DOC] SuperAnnotate Vector JSON** — https://doc.superannotate.com/docs/vector-json — per-instance `createdBy` / `updatedBy` / timestamps. Lowest-friction field names to mirror.
- **[DOC] Label Studio export** — https://labelstud.io/guide/export — `completed_by`, plus stable region IDs that persist from model prediction → human correction, giving you a correction trail for free.

---

## 2. Labels: worm, eggs (and "eggs = interpolation?")

- **Peng et al. 2025, bioRxiv — From manual counting to YOLO** — https://www.biorxiv.org/content/10.1101/2025.08.22.671778v1
  YOLOv11 egg counting trained on **only 331 annotated images**: 92% recall, 94.6% precision, MAE 0.95 eggs/image vs 2.65 for humans. Also documents human counters drifting over six months. Your annotation budget is smaller than you think.
- **Ji et al. 2024, *Genetics*** — https://pmc.ncbi.nlm.nih.gov/articles/PMC11631399/
  Mask R-CNN egg detection (P 0.85 / R 0.81) in 96-well plates, correlated against locomotor rhythm. Instance masks rather than boxes, and a realistic accuracy target.
- **Banse et al. 2024, *Lab on a Chip* — The egg-counter** — https://doi.org/10.1039/D3LC01073B
  Read this for the **GUI design**, not the microfluidics: a validated *propose-then-confirm* annotation loop where software suggests candidate eggs and the human accepts/rejects. That is how annotation stops being drawing.
- **Ravi et al. 2024 — SAM 2** — https://arxiv.org/abs/2408.00714
  The real answer to "eggs = interpolation?": don't linearly interpolate. Prompt once, let the memory bank propagate the mask through the video, correct where it drifts. Linear interpolation assumes constant velocity and breaks on eggs that are laid, sit still, then get occluded.
- **Geng, Cosman, Palm & Schafer 2005, *EURASIP J. Adv. Signal Process.*** — https://doi.org/10.1155/ASP.2005.2229
  Classical baseline for detecting egg-laying *events* (a temporal act) rather than counting deposited eggs (static objects). Worth distinguishing these two label types in your schema.

---

## 3. Program ændringer — +/- prompts, micro-SAM, editing after "segment all"

- **Archit et al. 2025, *Nat Methods* — Segment Anything for Microscopy** — https://doi.org/10.1038/s41592-024-02580-4 · preprint https://www.biorxiv.org/content/10.1101/2023.08.21.554208v1
  Hits three of your bullets at once. µSAM is a napari plugin with automatic segmentation *and* interactive correction, and its training loop simulates the correction process with point prompts, box prompts, and the previous mask's logits fed back to the decoder.
- **[DOC] micro-sam documentation** — https://computational-cell-analytics.github.io/micro-sam/micro_sam.html
  Annotator 2D / 3D / **Tracking**, the automatic-segmentation modes (AMG / AIS / APG), loading segmentations made by *another tool* into the `committed_objects` layer for correction, and a graphical finetuning UI.
- **Ravi et al. 2024 — SAM 2** — https://arxiv.org/abs/2408.00714
  Read §3 and the interactive-refinement protocol. The part most people miss: `add_new_mask` accepts a **full binary mask as a prompt on any frame**, so an existing worm mask can re-anchor a track. That is your "kan segmenteringen bruges som prompt?" — yes.
- **SAMRefiner: Taming SAM for Universal Mask Refinement (ICLR 2025)** — https://arxiv.org/abs/2502.06756
  Mask-in → better-mask-out as a standalone step: auto-derives a noise-tolerant prompt set (points + box + distance-guided cues) from a coarse mask, with split-then-merge for multi-object scenes.
- **Mazurowski et al. 2023, *Medical Image Analysis* — SAM: an experimental study** — https://arxiv.org/abs/2304.10517 · https://pmc.ncbi.nlm.nih.gov/articles/PMC10528428/
  Empirical basis for your UI defaults: box vs single point vs iterative clicks across 19 datasets, how much each extra click actually buys, and where more points *hurt*. Boxes tend to win for compact elongated objects.
- **Guisnet et al. 2026, *PLOS Comp Biol* — TWARDIS** — https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013441
  The closest published analogue to wormscope: SAM for static worm images + SAM2 for video, fine-tuned ViT classifier, morphometrics at r = 0.999 vs manual, handles overlapping worms. Read before you re-solve their problems.
- **Liu et al. 2023, ICCV — SimpleClick** — https://arxiv.org/abs/2210.11006
  Where the interactive paradigm comes from, and — useful for you — **NoC@IoU** (number of clicks to reach target quality). That is the metric that proves your correction tool saves annotator time.
- **Kirillov et al. 2023, ICCV — Segment Anything** — https://arxiv.org/abs/2304.02643
  Background but load-bearing: why SAM returns three masks, what `mask_input` is for, and the assisted → semi-automatic → fully-automatic data-engine ladder your workflow should climb.
- **Chen et al. 2025 — Segment Anything for Cell Tracking** — https://arxiv.org/abs/2509.09943
  Zero-shot SAM2 tracking through dense clusters and weak boundaries — i.e. your worst case, worms crossing.

**Code to read:**
SAM2 https://github.com/facebookresearch/sam2 (`add_new_mask`, `add_new_points_or_box`) ·
micro-sam https://github.com/computational-cell-analytics/micro-sam ·
napari SAM plugins as UI prior art: https://github.com/MIC-DKFZ/napari-sam · https://github.com/FrancisCrickInstitute/napari_sam2 · https://github.com/hiroalchem/napari-SAM4IS

---

## 4. Deployment package

**The evidence points away from Docker for a PyQt6 desktop app.**

- **Ouyang et al. 2023 — Moving beyond the desktop** — https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2023.1233748/full
  ~7–8% of recent image.sc posts contain the word "install"; installation appears in roughly 1 in 5 threads. Move the burden from user to developer.
- **[DOC] micro-sam install docs** — https://computational-cell-analytics.github.io/micro-sam/micro_sam.html
  The closest comparable project marks its **standalone installers as deprecated** and recommends `conda create -c conda-forge -n micro-sam micro_sam`, because conda resolves the right PyTorch build. Strong prior.
- **List 2017, *PLOS Comp Biol* — Ten simple rules for developing usable software** — https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1005265
  Rule 8 "get users started quickly" is your to-do item verbatim; Rules 6–7 cover logging versions/parameters and expecting user mistakes.
- **Taschuk & Wilson 2017 — Ten simple rules for making research software more robust** — https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1005412
  "Bullet proof" as a checklist: version your releases, use package managers, never require root, no hard-coded paths, ship a small test set.
- **Grüning et al. 2018, *F1000Research* — Packaging and containerizing bioinformatics software** — https://f1000research.com/articles/7-742/v1
  Package first (conda recipe), container second — the container falls out of the recipe automatically.
- **[DOC] Pawsey — GUI applications in containers via X11** — https://pawseysc.github.io/singularity-containers/42-x11-gnuplot/index.html
  The honest limitation: X server on the host, XQuartz on macOS, `xauth` inside the image, `-B ~/.Xauthority`. Four things must go right on your colleague's machine. Verdict: don't ship a PyQt6 app this way.
- **Sun, Hériché & Kutra 2024 — BAND: a cloud-based virtual desktop for bioimage analysis** — https://zenodo.org/doi/10.5281/zenodo.12699363 · live: https://band.embl.de/
  How the field actually solves "GUI in a container": a hosted remote desktop, not a local container.
- **Moreau & Wiebels 2026 — Nine quick tips for software containerization** — https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1014197
  Tip 1: clarify *why* you're containerizing before you write a Dockerfile. Tip 5: keep data outside the image.
- **Nüst et al. 2020 — Ten simple rules for writing Dockerfiles** — https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1008316 — if you do containerize anyway.
- **Kurtzer et al. 2017 — Singularity** — https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0177459 — rootless containers, for IT-managed university machines where Docker Desktop is blocked.
- **Levet et al. 2021, *F1000Research* — Developing open-source software for bioimage analysis** — https://pmc.ncbi.nlm.nih.gov/articles/PMC8226416/ — the community's stated norm: 1-click installers on all three OSes where feasible.
- **Jamali et al. 2021 — 2020 BioImage Analysis Survey** — https://www.cambridge.org/core/journals/biological-imaging/article/2020-bioimage-analysis-survey-community-experiences-and-needs-for-the-future/9E824DC0C27568FE5B9D12FB59B1BB90 — 484 imaging professionals; the top request is **better documentation and tutorials**, ahead of the installer itself.

**[DOC] Tooling:** pixi https://pixi.sh/latest/ · uv lock/sync https://docs.astral.sh/uv/concepts/projects/sync/ · pyproject.toml https://packaging.python.org/en/latest/guides/writing-pyproject-toml/ (use `[project.gui-scripts]`, not `[project.scripts]`, or Windows opens a console window behind the Qt window) · conda constructor (1-click installers, how napari ships) https://conda.github.io/constructor/ · conda-forge recipes https://conda-forge.org/docs/maintainer/adding_pkgs/ · PyInstaller https://pyinstaller.org/en/stable/ (not a cross-compiler — needs CI on each OS) · Briefcase https://briefcase.beeware.org/en/stable/

### "Gemme dataen?" — this is a five-line fix

- **[DOC] Qt `QSaveFile`** — https://doc.qt.io/qt-6/qsavefile.html — writes to a temp file, `commit()` moves it into place, detects a full partition and discards rather than truncating. Available in PyQt6.
- **[DOC] Moyer 2011, LWN — Ensuring data reaches disk** — https://lwn.net/Articles/457667/ — the full recipe: temp file on the same filesystem → write → `fsync` the file → rename over target → **`fsync` the directory** (skipping the last step is the classic "file exists but is empty after a crash").
- **Hart et al. 2016 — Ten simple rules for digital data storage** — https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1005097 — the policy layer: raw data never modified, derived outputs separate, dated backups, open formats.
- **[DOC] pytest-qt** — https://pytest-qt.readthedocs.io/en/latest/ — `qtbot` in headless CI. Note "Exceptions in virtual methods": Qt swallows Python exceptions in virtual methods, so a crash looks like "the button did nothing".
- **[DOC] GitHub Actions for Python** — https://docs.github.com/en/actions/tutorials/build-and-test-code/python — matrix over ubuntu/macos/windows, so you find out her Windows install is broken before she does.
- **[DOC] Semantic Versioning** — https://semver.org/ — and write `__version__` into every saved annotation file.

### The napari question you should cost out before building more

- **[DOC] napari plugins (users)** — https://napari.org/stable/plugins/index.html — a biologist clicks Install in a dialog. That is a deployment package you don't have to build.
- **[DOC] Your first plugin (npe2)** — https://napari.org/stable/plugins/building_a_plugin/first_plugin.html — napari *is* a Qt app, so your PyQt6 widgets can become dock widgets rather than a rewrite; you inherit layers, time slider, label painting, undo.
- Citation note: napari has no journal paper. Cite the Zenodo record https://doi.org/10.5281/zenodo.3555620 — not an invented "napari, Nature Methods".

---

## 5. Robust model — finetuning, micro-SAM, napari

- **Teuber, Archit & Pape 2025 — Parameter-efficient finetuning of SAM for biomedical imaging** — https://arxiv.org/abs/2502.00418 · code https://github.com/computational-cell-analytics/peft-sam
  From the micro-sam group: LoRA vs adapters vs BitFit vs layer-norm vs **decoder-only** vs full finetuning, specifically in the small-dataset regime. Decoder-only is a strong baseline.
- **Gu, Dong, … Mazurowski 2024/2025 — How to build the best medical segmentation algorithm using foundation models** — https://arxiv.org/abs/2404.09957 · code https://github.com/mazurowski-lab/finetune-SAM
  18-variant ablation over where to inject adaptation and what prompt regime to train under — including when a plain U-Net still wins. Read before spending GPU hours.
- **Lightweight open-source fine-tuning of SAM2 for domain-specific microscopy segmentation, bioRxiv 2025** — https://www.biorxiv.org/content/10.1101/2025.11.08.687405v1 · https://pmc.ncbi.nlm.nih.gov/articles/PMC12642685/
  A copy-able SAM2 (not SAM1) finetuning recipe for modest hardware and small annotated sets. *Caveat: title/DOI/PMC record confirmed, author list not verified — check the byline before citing.*
- **Ma, Wang et al. 2025 — MedSAM2** — https://arxiv.org/abs/2504.03600 · https://github.com/bowang-lab/MedSAM2
  A shipped precedent for "finetuned SAM2 + correction UI", including human-in-the-loop studies that quantify the manual-effort reduction.
- **Deserno & Bozek 2023, *Sci Rep* — WormSwin** — https://doi.org/10.1038/s41598-023-38213-7 · **data:** https://doi.org/10.5281/zenodo.7456803
  Free training data: 60,500+ annotated C. elegans instance masks (CSB-1), 10,000 synthetic images, ~3,900 mating/overlap postures, plus BBBC010 benchmarking. Start here instead of annotating from scratch.
- **Pachitariu, Rariden & Stringer 2025 — Cellpose-SAM** — https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1
  The alternative for the *automatic* pass, and a proven HITL pattern to mirror: correct a few masks → retrain in seconds → re-run.
- **Colomer et al. 2026, *J. Imaging* 12(7):297 — Microscopy cell segmentation: review and benchmarking** — https://www.mdpi.com/2313-433X/12/7/297
  StarDist vs CellSAM vs Cellpose-SAM vs detector→box-prompt→SAM. That last pattern (auto-boxes instead of a point grid) is a plausible upgrade to your "segment all".
- **Minimal finetuning code:** https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code — smallest readable SAM2 training loop; good first experiment.

---

## 6. Pulsation — data, how it's done now, strategy

- **Shappell et al. 2026, *PLOS Comp Biol* — PumpKin** — https://doi.org/10.1371/journal.pcbi.1014489 · https://github.com/lu-lab/PumpKin
  Two chained Faster R-CNN detectors: find the pharyngeal bulb → emit a dynamic ROI → track the grinder inside the crop. Then motion compensation, sparse optical flow, low-pass Butterworth, and a biphasic peak detector. 69% agreement with manual annotators — where annotators only agree with *each other* 59%. Keep that second number; it reframes "69%" from failure to success.
- **Bonnard et al. 2022, *eLife* — PharaGlow** — https://elifesciences.org/articles/77252
  Segment the pharynx → extract centerline/width → **virtually straighten** along the AP axis → per-frame scalar = inverted max of the dorso-ventral SD of brightness → AMPD peak detection with physiological interval limits. 30–50 freely moving animals at once.
- **Scholz et al. 2016, *J. Neurosci. Methods*** — https://www.biorxiv.org/content/10.1101/051243v1
  The cheapest thing that works, ~30 lines: difference two frames 16 ms apart, compute the **entropy** of the intensity-difference distribution, peak-detect. No training data, no fluorophore. Try this first.
- **Cermak et al. 2020, *eLife*** — https://elifesciences.org/articles/57093
  Extracts pumping, defecation **and egg-laying** simultaneously from brightfield video over hours to days. Resolves the pharyngeal-vs-defecation ambiguity and bridges your pulsation and egg sections.
- **Ferreira et al. 2024, *Bioinformatics* — FEHAT** — https://doi.org/10.1093/bioinformatics/btae664 · https://github.com/birneylab/FEHAT
  The clean FFT pipeline, plus the trick that finds the pulsating organ for you: rolling absolute frame difference → triangle threshold → the periodically-varying pixels *are* the organ. Includes explicit second-harmonic correction (the classic 2× rate error).
- **Xing et al. 2018, *Med Biol Eng Comput*** — https://doi.org/10.1007/s11517-018-1863-7
  Heart rate from **3–4 fps** zebrafish video via dense multiresolution optical flow + PCA — it never sees the heart, only the deformation it causes. Your fallback when nothing crisp is trackable.
- **Kim 2011, *Healthc Inform Res*** — https://pmc.ncbi.nlm.nih.gov/articles/PMC3155168/
  Benchmarks the full 2×2 of {optical flow, difference imaging} × {peak detection, FFT}. Optical flow + peak detection won. Evidence instead of a coin flip.
- **Lauridsen et al. 2019, *BMC Biology* — Eulerian video magnification** — https://pmc.ncbi.nlm.nih.gov/articles/PMC6907275/
  Amplify sub-visible pulsation so a human (or a detector) can actually see it. Use as an annotation aid.
- **Santoso et al. 2020, *Animals* — Daphnia cardiac performance in ImageJ** — https://pmc.ncbi.nlm.nih.gov/articles/PMC7552287/
  Reproducible today: ROI over the pulsating chamber + ImageJ Time Series Analyzer + peak detection, using inter-peak intervals so beat-to-beat variability survives. Good manual ground truth.
- **Dillon et al. 2009, *PLoS ONE* — AutoEPG** — https://pmc.ncbi.nlm.nih.gov/articles/PMC2795780/
  Electropharyngeogram = the electrical ground truth for pump timing, and the ceiling any video method is measured against.
- **Raizen et al. 2018, WormBook — Methods for measuring pharyngeal behaviors** — https://www.ncbi.nlm.nih.gov/books/NBK126648/
  Read **before recording more data**: ≥10 fps minimum for video scoring, replay at ⅓–½ speed, plate animals individually to reduce variance.

### Strategy that the literature converges on

1. Straighten first. Use the segmentation you already have to get a centerline and work in a body-fixed frame — this removes locomotion and bending as confounds.
2. Find the pulsating structure by **temporal variance, not appearance** (rolling frame difference + threshold). Zero training data.
3. Reduce to **one scalar per frame**. The scalar is the real design choice: entropy of frame difference (cheapest), inverted max of dorso-ventral brightness SD, summed ROI intensity, or tracked grinder displacement (most work).
4. Two-stage detection beats one-stage for small structures: detect the big landmark → crop → detect inside the crop.
5. FFT gives a robust *average rate* (watch the second harmonic); peak detection gives *event times*, hence intervals and variability. Compute both and use the spectrum to sanity-check the peak count.
6. Band-limit and constrain physiologically: Butterworth low-pass, baseline-drift removal, minimum peak height, minimum inter-event interval. Most false positives die here.
7. Sample well above 2× the expected rate. No algorithm recovers from aliasing.
8. Report inter-annotator agreement alongside model-vs-human agreement, or your numbers look worse than they are.

---

## 7. Metrics — speed, acceleration, area, length

- **Yemini et al. 2013, *Nat Methods* — A database of C. elegans behavioral phenotypes** — https://pmc.ncbi.nlm.nih.gov/articles/PMC3962822/
  The canonical definitions. **Speed** = centroid displacement between start and end frames of a 1/4–1 s window, computed separately for head tip, head, midbody and tail. **Length** = chain-code pixel length of the segmented skeleton, converted to µm. **Area** = pixel count inside the segmented contour. **Path curvature** = path angle in radians / distance travelled in µm.
- **Javer et al. 2018, *Phil Trans R Soc B* — Tierpsy features** — https://pmc.ncbi.nlm.nih.gov/articles/PMC6158219/
  How the ~3000-feature space is built from skeletons, and how it reduces to interpretable subsets (Tierpsy-256, Tierpsy-16) without losing classification power. Use this to pick the handful you surface in the GUI.
- **Vedantham et al. 2025, *eNeuro* — Track-A-Worm 2.0** — https://www.eneuro.org/content/12/8/ENEURO.0224-25.2025
  Speed, distance, direction, amplitude, length, curvature, dominant bending frequency — plus a citable "is it actually moving" threshold (<10 µm inter-frame displacement) for separating real motion from segmentation jitter.
- **Wang & Wang 2013, *PLOS ONE* — Track-A-Worm** — https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0069653
  Small, readable MATLAB reference implementation of skeleton→metric maths.
- **Husson et al. 2012, WormBook — Keeping track of worm trackers** — https://www.wormbook.org/chapters/www_tracking/tracking.html
  Fast orientation to the whole tool landscape plus the shared vocabulary. Read first if you want the map.
- **O'Brien et al. 2025, *eLife*** — https://elifesciences.org/articles/92491 — a worked example of validating that features actually separate conditions.
- **[DOC] Tierpsy "Software Explanation"** — https://mgh17.github.io/tierpsy-tracker/docs/EXPLANATION.html
  The highest-value doc here: compression → trajectory linking → SegWorm skeletonization (head/tail = highest-curvature contour points; skeleton = midpoints between opposite contour sides) → head/tail orientation → features.

### Three traps

- **Don't finite-difference per frame.** Yemini's speed is displacement over a 1/4–1 s window. Per-frame differencing on segmentation jitter inflates speed and destroys acceleration. Acceleration is the one metric on your list that nobody defines canonically — you'll derive it, so smooth deliberately and document the window.
- **Length from a thinned mask is biased short**, and the bias *varies with posture*, so it looks like real biology. Tierpsy uses contour midpoints and WormTracer explicitly optimises the endpoints for exactly this reason.
- **Coiled frames are not noise.** Log them rather than dropping them silently — see below.

### Skeletonization pitfalls, coiling, omega turns

- **Broekmans et al. 2016, *eLife*** — https://elifesciences.org/articles/17227 — the argument that discarded coiled frames contain real biology: "omega turns" turned out to be two distinct behaviours.
- **Kuze et al. 2023/2025 — WormTracer** — https://www.biorxiv.org/content/10.1101/2023.12.11.571048v1 — uses temporal continuity (postures before and after a coil) plus endpoint optimisation. Read if your lengths jitter or shrink during turns.
- **Nagy et al. 2015, *PLOS Comp Biol*** — https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1004517 — model-based coil resolution with no training set and no GPU.
- **Roussel et al. 2014, *Worm*** — https://doi.org/10.4161/21624054.2014.982437 — the failure taxonomy: coiling, entanglement, omega bends. A checklist to test your skeletonizer against.
- **Hebert et al. 2021, *PLOS Comp Biol* — WormPose** — https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1008914 — CNN trained on *synthetic* worm images (no manual labels) to recover centerlines where classical skeletonization fails; designed to sit downstream of Tierpsy.
- **Alonso & Kirkegaard 2023, *Commun Biol* — DeepTangle** — https://doi.org/10.1038/s42003-023-05098-1 — predicts centerlines *directly* instead of skeletonizing masks, ~90 Hz, trained purely on synthetic data.
- **Weheliye et al. 2025, *PLOS Comp Biol* — DeepTangleCrawl** — https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013345 — the crawling-worm version, with evidence that trajectory continuity improves screen signal-to-noise.

### Keeping identity through collisions (needed before acceleration means anything)

- **Winter et al. 2016, *Sci Rep* — WALDO** — https://doi.org/10.1038/srep34859 — **the worm-specific collision paper.** Fragmented tracks as a directed acyclic network; collisions resolved by pixel-overlap comparison of pre- and post-collision blobs. Start here, not with generic MOT.
- **Romero-Ferrero et al. 2019, *Nat Methods* — idtracker.ai** — https://arxiv.org/abs/1803.04351 — identity by classification, not association; up to ~100 unmarked individuals.
- **Walter & Couzin 2021, *eLife* — TRex** — https://elifesciences.org/articles/64000 — posture *and* identity, 2.5–46.7× faster than idtracker.ai.
- **Zhang et al. 2021 — ByteTrack** — https://arxiv.org/abs/2110.06864 — don't discard low-confidence detections; a worm mid-coil produces exactly the low-score box naive trackers throw away.
- **Bewley et al. 2016 — SORT** — https://arxiv.org/abs/1602.00763 — ~300 lines, worth reading in full to see which motion assumptions worms violate.
- **Liu et al. 2025, *PLOS Comp Biol*** — https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013707 — YOLOv8 + ByteTrack on worms: 99.5% precision, 153 FPS, occlusion handling as an explicit goal.

---

## 8. Interactions + heatmap

- **Ding et al. 2019, *eLife* — Shared behavioral mechanisms underlie C. elegans aggregation and swarming** — https://doi.org/10.7554/eLife.43318
  Your species, your question. Pair-correlation functions, hierarchical clustering of positions, SD and kurtosis of the position distribution as compactness, density-dependent reversal rates, neighbour-density-dependent speed — all implementable directly on wormscope masks.
- **Farine & Whitehead 2015, *J Anim Ecol* — Constructing, conducting and interpreting animal social network analysis** — https://doi.org/10.1111/1365-2656.12418
  **Read the association-definition and permutation-test sections before you threshold "distance < X px" into edges.** Standard statistics are invalid on network data; you need permutation nulls. R code in the appendices.
- **Romero-Ferrero et al. 2019 — idtracker.ai** — https://doi.org/10.1038/s41592-018-0295-5 — its touching/crossing detector is itself an interaction detector.
- **Lauer et al. 2022, *Nat Methods* — Multi-animal DeepLabCut** — https://doi.org/10.1038/s41592-022-01443-0 — posture-level interaction measures (head-to-tail contact) and occlusion recovery.
- **Worton 1989, *Ecology* — Kernel methods for the utilization distribution** — https://doi.org/10.2307/1938423 — your occupancy heatmap is a utilization distribution; this is why KDE beats a raw 2D histogram, and how to choose the bandwidth.
- **Fleming et al. 2017, *Methods Ecol Evol* — Autocorrelated KDE** — https://doi.org/10.1111/2041-210X.12673 — **the correction that makes the above valid on video.** Per-frame positions are massively autocorrelated, which collapses classical bandwidth selection and biases occupancy area.

---

## 9. Hvordan dele det annoterede data?

- **Zulueta-Coarasa et al. 2025, *Nat Methods* — MIFA** — https://doi.org/10.1038/s41592-025-02835-8
  The one to read: metadata modules describing the *annotations themselves*, recommended formats (OME-Zarr), explicit Creative Commons licensing, deposition to BioImage Archive / EMPIAR. Written for exactly your case — a human-annotated microscopy dataset meant for model training.
- **Sarkans et al. 2021, *Nat Methods* — REMBI** — https://doi.org/10.1038/s41592-021-01166-8
  The metadata checklist repositories actually enforce: study-level, image-data, and *analyzed-data* modules. Capture these fields **while annotating**, not a year later at deposition.
- **[DOC] BioImage Archive** — https://www.ebi.ac.uk/bioimage-archive/ · paper: Hartley et al. 2022, *J Mol Biol* https://doi.org/10.1016/j.jmb.2022.167505
  The default home for raw + annotated life-science image data, no size ceiling.
- **Williams et al. 2017, *Nat Methods* — Image Data Resource** — https://doi.org/10.1038/nmeth.4326 — the curated, browsable counterpart, for datasets others should reanalyse in place.
- **Wilkinson et al. 2016, *Sci Data* — FAIR principles** — https://doi.org/10.1038/sdata.2016.18 — the canonical citation. The R (Reusable) principle ties your provenance fields to deposition.
- **Bajcsy et al. 2025, *Nat Methods* — Enabling global image data sharing** — https://doi.org/10.1038/s41592-024-02585-z — current map: ~6 PB across repositories, and the practical obstacles.
- **Gebru et al. 2018/2021 — Datasheets for Datasets** — https://arxiv.org/abs/1803.09010 — ready-made template for the human-readable document that ships with the data: annotator instructions, inter-annotator disagreement, known label limitations.
- **Javer et al. 2018 — Tierpsy/WCON + community database** — https://doi.org/10.1038/s41592-018-0112-1 — the worm-community-specific sharing route.

---

## Two gaps worth knowing about

- **There is no standard for annotation versioning / correction history in bioimaging.** The nearest working answers are the tool schemas above (SuperAnnotate field names, Label Studio stable region IDs), backed by OMERO's ownership model. If wormscope invents its own, mirror SuperAnnotate's names.
- **Acceleration has no canonical definition** in the worm literature — every metric in your list except that one comes from Yemini 2013. You will be defining it, so pick a smoothing window deliberately and write it down.
