# Chapter 15: Dynamic clustering for synchrony features

> **Delayed (18 November 2026).** The clustering experiment is shelved behind experiment 09 (visual
> focus of attention). Head direction from 09 is the natural input for the synchrony features here:
> the classroom synchrony in the literature (Raca et al.) is head-rotation synchrony. The options
> below stand; the experiment will be renumbered when it is picked up.

FYP-1's first claim is that cheap features can measure attention well enough. The cheap features
it rests on are *class-relative*: how one student behaves compared with the students around them,
in the same window. This chapter covers:

- why such features are worth trying;
- the constraints our data imposes;
- every clustering family we considered, and what each would give us;
- which ones the clustering experiment tests, and how they plug into the code.

The pipeline's CFMU extractors (§15.7) are not built yet. The features themselves, and the scripts
that decide which ones to use, are in `src/cfmu/classrel.py` and `scripts/clustering/` (§15.9).
The CFMU slot they build on is described in Chapter 6 §6.6, and the spread maths in
`src/cfmu/spread.py`. Status as of 23 November 2026.

## 15.1 Why class-relative features

**What the literature reports.** Three findings from classroom studies:

- **Raca, Tormey and Dillenbourg (LAK 2014; *Journal of Learning Analytics* 2016)** recorded
  classrooms on camera.
  - A student's raw amount of movement did not predict their self-reported attention.
  - Their *timing* relative to the class did: less attentive students react later than their
    peers (the "sleepers' lag").
  - The share of near-synchronous head movements tracked the class's attention.
- **Goldberg et al. (*Educational Psychology Review*, 2019)** estimated visible engagement from
  gaze, head pose and facial expression. Adding a synchrony feature improved the correlation with
  manual ratings and the prediction of post-test results.
- **Dikker et al. (*Current Biology*, 2017)** measured synchrony in EEG rather than video. Across
  students it tracked class engagement. This is evidence that "moving with the class" reflects
  shared attention, not just shared posture.

**Why this suits our data.** Experiment 06 and the feature audit (experiment 08's motivation) found
two problems that class-relative features address directly.

1. **Our features describe the seat and the camera; the labels change within a student.**
   - Between 37% and 78% of the variance of the motion and flow features lies between tracks.
   - Only 33% of labeler_01's rating variance lies between students (06 §1.7).
   - A feature measured *relative to the other students in the same frame* cancels most camera,
     distance and lighting effects. They affect the student and their classmates alike.
2. **The class-context columns act as session identifiers.** The current context (`use_class_context`,
   68 `vec.*` columns) is the same for every student in a window, so it can only tell sessions
   apart. A class-relative feature differs between the students of one window. It is context
   expressed from the student's point of view.

**What "dynamic" means here.** Every window has its own set of visible students. The groups are
formed again in each window, from what the students are doing in it. The clustering follows the
class rather than being fixed in advance.

## 15.2 The setting in numbers

**Visible students per window** (52 original session-cameras, 4 s grid, cache
`outputs/cache/features/2685b0745fde`; a track counts if it covers at least half the window,
`vec.py:39-47`):

| Students | 0 | 7–10 | 11–14 | 15–17 | 18–20 |
|---|---|---|---|---|---|
| Windows | 648 (8.4%) | 1,163 (15%) | 3,225 (42%) | 2,387 (31%) | 263 (3.4%) |

- The median is 12 and the maximum 20.
- Apart from the empty windows (camera gaps), a window never has fewer than 7 students.
- The **small-n** regime is n ≈ 7–20, never hundreds. This rules out methods that need many points
  per cluster (§15.5).

**What each student has in a window:**

| Kind | Where | Shape per student | Content |
|---|---|---|---|
| Per-frame signals | `outputs/cache/tracks/<tracking hash>/<s>.npz` | about 20 frames (about 5 fps × 4 s) × 13 channels, plus box and velocity | residual-flow cells `rf.mag_ij`, SCB scores (`scb.hand_raising`, `read`, `write`, `m0.obj`), Kalman velocity, box |
| Window descriptors | the feature pass (`trk.*`); the label-window cache | 27 numbers | Kalman geometry (10), flow cells (9), SCB mean and peak (8) |

All tracks of one camera share the same frame timestamps, so per-frame signals line up across
students without resampling. A student missing from some frames (occlusion) is compared on the
frames both have.

**Caveat carried over from the audit.**
- The motion channels sit near the noise floor: median Kalman speed is about 0.08 px per frame, and
  median residual flow about 0.05 px per frame at half resolution.
- Correlating two noise signals gives a correlation near 0, so synchrony features are only
  informative on signals above that floor.
- The clustering experiment therefore runs on improved features (head direction from experiment 09),
  not on 06's.

## 15.3 Requirements

Any method we use has to satisfy:

| # | Requirement | Why |
|---|---|---|
| R1 | Permutation invariance | Tracker ids are arbitrary; reordering students must not change any feature. |
| R2 | Variable n, with defined NaN rules | n changes every window (0, then 7–20). Every key must exist in every window, NaN where undefined (the Vec contract, §6.4). |
| R3 | No fixed number of clusters | With 7–20 students, a fixed K is wrong half the time: the class may be doing one thing, or several. |
| R4 | Stable over time | Cluster ids change arbitrarily between windows. A feature must not depend on the id ("cluster 2"), only on relations ("in the largest group", "an outlier"). |
| R5 | Works at small n | The statistics are over 7–20 points; methods that estimate densities or covariances per cluster have little to work with. |
| R6 | Removes the seat confound | Descriptors are standardised per track (over the session) before clustering, so the method groups what students do, not where they sit or how far they are from the camera. |
| R7 | No label leakage | Anything fitted is fitted without labels. If it is fitted across sessions, the leakage argument is stated (§15.4, option 4). |
| R8 | Cheap | FYP-1's claim is about cheap features. It must cost far less than the frame pass (about 185 ms per frame). |

## 15.4 The options

Two choices come before the algorithm: **what is clustered** and **in which space**.

- **Descriptor space:** each student is one point, their window descriptor (27 numbers, or a smaller
  chosen subset). "Similar" means doing similar things on average over the window.
- **Synchrony space:** each student is a time series over the window's frames (for example their
  total residual flow, or `scb.read`). "Similar" means rising and falling together. The distance is
  1 − Pearson correlation on the common frames. This is the space the sleepers'-lag finding
  is about.

Every option below works in either space, unless noted.

### Option 0: no clustering, the student against the rest of the class

The reference point every clustering method has to beat. For student *i* in a window with *n*
visible students:

- **Deviation from the class:** dev_i = median over d of |x_id − med_{j≠i} x_jd| / MAD_d.
  - The leave-one-out median of each descriptor dimension, divided by that dimension's spread
    across the class (the median absolute deviation, MAD).
  - Robust to a single outlier, and defined for n ≥ 3.
- **Synchrony with the class:** sync_i = corr(s_i(t), m_{−i}(t)), where m_{−i} is the median of
  the other students' series frame by frame.
- **Lag behind the class:** lag_i = argmax over k of corr(s_i(t), m_{−i}(t − k)), for |k| up to
  about 1 s.
  - A direct reading of the sleepers' lag.
  - At about 5 fps a 4 s window has 20 frames, so lag estimates are coarse. 8 s windows (the 8 s
    features are already cached) give more frames.
- **Graph degree:** the student's mean pairwise correlation with every other student.
- **Window level:**
  - the class's mean pairwise correlation (overall synchrony);
  - the share of students whose sync_i exceeds a threshold.

| | |
|---|---|
| Variable n | Defined for any n ≥ 3, NaN below. |
| Cost | O(n² T) per window: negligible at n ≤ 20, T ≈ 20. |
| Failure modes | On noise-floor signals every correlation is near 0. With few common frames correlations are unstable, so require at least 8. |
| Requirements met | R1–R8 by construction; R3 and R4 do not arise (no clusters). |

If option 0 does as well as the clustering methods, the clustering is not justified. That outcome is
a valid result for the clustering experiment.

### Option 1: agglomerative clustering with a distance threshold

- **How it works:** hierarchical clustering (`sklearn.cluster.AgglomerativeClustering`, or
  `scipy.cluster.hierarchy`). It starts with every student as their own cluster and merges the two
  closest until the closest pair is farther apart than a threshold τ.
- **Linkage:**
  - Ward (minimum within-cluster variance) in descriptor space;
  - average linkage in synchrony space (correlation distance is not Euclidean, so Ward is invalid
    there).
- **The number of clusters follows from τ (R3).** τ is set in standardised units, so it means the
  same in every session. It is tuned, or fixed at a value chosen on the training groups.
- **Features:**
  - **per student:**
    - the share of the class in the student's cluster (n_c / n);
    - a singleton flag ("does what nobody else does");
    - the distance to the cluster's centroid;
    - in synchrony space, whether the student is in the largest group.
  - **per window:**
    - the number of clusters divided by n;
    - the size of the largest cluster divided by n;
    - **chaos** and **fragmentation** from `spread_decomposition`: the within-cluster and
      between-cluster SDs, and the between-share per feature block (design §10.4, already
      implemented and tested).
- **Properties:** deterministic and defined for any n ≥ 2. The features depend only on group
  relations, not cluster ids (R4).
- **Cost:** O(n² log n) per window, negligible.
- **Failure modes:**
  - The result depends on τ: too small and everyone is a singleton, too large and there is one
    cluster.
  - Ward is sensitive to the outliers we most want to keep. Average linkage is less so.

### Option 2: density-based (HDBSCAN, DBSCAN)

- **How it works:** a cluster is a dense region; a point in no dense region is labelled *noise*.
  - **DBSCAN** (Ester et al., KDD 1996) needs a radius ε.
  - **HDBSCAN** (Campello, Moulavi and Sander, PAKDD 2013) builds the hierarchy over all radii and
    keeps the most stable clusters. It needs only `min_cluster_size`, and is in scikit-learn
    (`sklearn.cluster.HDBSCAN`, installed version 1.9.1).
- **Why it is attractive here:** the noise label is directly the feature we want: "this student fits
  no group of classmates".
  - scikit-learn also returns a membership strength per point (`probabilities_`), a soft version
    of the same signal.
  - The GLOSH outlier score from the paper's journal version needs the separate `hdbscan`
    package, which is not installed.
- **Features:**
  - per student: noise flag, membership strength, cluster share;
  - per window: noise share, number of clusters, chaos and fragmentation (as in option 1).
- **Failure modes at our n:**
  - With n = 7–20, `min_cluster_size` must be 2 or 3. HDBSCAN's stability selection then has few
    levels to choose from, and the clustering can flip between "one cluster" and "all noise".
  - Set `allow_single_cluster=True`, so that a class all doing the same thing is one cluster, not
    noise.
- **Cost:** negligible at our n.

### Option 3: model-based with an automatic number of clusters (GMM with BIC, Dirichlet-process mixture)

- **How it works:** fit Gaussian mixtures and choose the number of components.
  - Either fit each K and keep the lowest BIC;
  - or fit a Dirichlet-process mixture (`sklearn.mixture.BayesianGaussianMixture`, variational
    inference) that switches off unused components.
- **Features:** soft memberships, so the per-student feature is the entropy of the membership
  (high means between groups) and the expected cluster share.
- **Why not, at our n:**
  - A covariance per component needs far more than 7–20 points in 27 dimensions.
  - Even diagonal covariances with strong priors give memberships that depend on the prior more
    than on the data.
  - Reducing to 2–4 dimensions first (PCA within the window) helps but adds another fragile step.
- **Verdict:** out for the clustering experiment. The soft-membership idea survives in option 4, where the
  mixture is fitted once on many points.

### Option 4: a global prototype codebook (behaviour states shared by all sessions)

- **How it works:** fit k-means or a diagonal GMM with K ≈ 6–10 prototypes, once, on the
  descriptors of many students pooled over sessions (hundreds of thousands of track-windows).
  - In each window, every student gets a soft assignment to the fixed prototypes.
  - The prototypes act as a vocabulary of behaviour states, such as "still, looking down" or
    "moving, hand up".
- **Why it fits our constraints:**
  - **It solves R4 completely.** A prototype means the same thing in every window and session, so
    "what share of the class is in my state" is comparable everywhere.
  - **It handles variable n trivially** (R2, R5): nothing is estimated within the window.
- **Features:**
  - **per student:**
    - **conformity**: the share of the class in the student's prototype, or with soft
      assignments, the overlap between the student's assignment and the class mean assignment;
    - the entropy of the student's own assignment;
    - the distance to the nearest prototype.
  - **per window:**
    - the class's prototype distribution (K numbers);
    - its entropy (low means everyone in one state).
- **Leakage (R7):** the codebook uses no labels. Two ways to fit it:
  1. On all original sessions (transductive, label-free: the held-out group's descriptors shape the
     prototypes, but not its labels). This is standard for unsupervised features, but has to be
     stated.
  2. Within each fold on the training groups only, which is stricter. The descriptors already sit
     in the cached per-track rows (`trk.*` in `<s>.labelwin.csv` and `<s>.tracks.csv`), so this
     can run inside `train_models` per fold with no re-extraction.

  We use (2).
- **Cost:** one k-means fit per fold (seconds), then O(nK) per window.
- **Failure modes:**
  - On noise-dominated descriptors the prototypes encode noise levels, which again means seat and
    distance. R6 (per-track standardisation) is essential.
  - K is a hyperparameter. Choose it on the training groups (BIC, or the held-out within-fold
    score).

### Option 5: temporal and graph methods

- **Evolutionary clustering** (Chakrabarti, Kumar and Tomkins, KDD 2006): each window's clustering
  trades fit to the current data against similarity to the previous window's clustering. Evolutionary
  versions of k-means and agglomerative clustering are given.
  - It would steady option 1 or 2 across our 50%-overlapping windows.
- **Matching clusters across windows** (MONIC: Spiliopoulou et al., KDD 2006): match each window's
  clusters to the previous window's by shared members (Jaccard overlap, Hungarian assignment). Then
  label the transitions: survive, split, merge, disappear, emerge.
  - Per-student features: how often the student changes group per minute; whether they leave the
    majority group before or after the class (a group-level lag).
- **Correlation graphs with community detection:** students are nodes, edges are positive
  pairwise correlations (synchrony space).
  - Communities via Louvain (Blondel et al. 2008; in networkx 3.7 as
    `nx.community.louvain_communities`) or Leiden (Traag, Waltman and van Eck, *Scientific Reports*
    2019; needs `igraph` and `leidenalg`, not installed).
  - The partition's modularity is a fragmentation measure.
  - The student's node strength equals option 0's graph degree.
- **Verdict:** the most expressive family, but each adds a state machine or a resolution parameter
  on top of a small, noisy n. Transitions also need the cluster sequence to be stable before they
  mean anything. Kept for FYP-2 (the design's "soft dynamic clustering"), unless options 1 and 4
  show the class-relative signal is there.

### Considered and ruled out

| Method | Why not |
|---|---|
| k-means or spectral clustering with a fixed K per window | violates R3; a fixed K forces splits that are not in the data, and cluster ids are not stable (R4) |
| Affinity propagation | chooses K, but through a "preference" whose effect at n ≈ 12 is erratic; no advantage over option 1's threshold |
| Mean-shift | needs a bandwidth, which at n ≈ 12 is the whole problem; its kernel density estimate is as unreliable as option 3's |
| Per-window GMM or DP mixture | option 3: too few points per window |
| Time-series clustering with DTW | DTW absorbs the timing differences that synchrony and lag are meant to measure |

## 15.5 Comparison

| | 0 Student vs class | 1 Agglomerative (τ) | 2 HDBSCAN | 3 Auto-K mixture | 4 Prototype codebook | 5 Temporal / graph |
|---|---|---|---|---|---|---|
| Fixed K needed (R3) | – | no (τ) | no (`min_cluster_size`) | no | K prototypes, but global | no (resolution) |
| Stable meaning over time (R4) | yes | relations only | relations only | relations only | **yes** | by design |
| Small n (R5) | yes | yes | fragile | **no** | yes | fragile |
| Soft memberships | – | no | strengths | yes | **yes** | no |
| Fitted across sessions | no | no | no | no | yes, per fold, label-free | no |
| Parameters | lag range, min frames | τ, linkage | min cluster size | priors | K | smoothness, resolution |
| Cost per window | O(n²T) | O(n² log n) | small | small | O(nK) | larger |
| Clustering experiment | **yes (the bar)** | **yes** | as a variant of 1 | no | **yes** | FYP-2 |

## 15.6 What the clustering experiment tests

**Arms**, each added to the experiment 08/09 configuration with the class-context columns off:

1. `null`: no class-relative features (the baseline).
2. `sync`: option 0. Deviation, synchrony and lag with the class, in descriptor space and on 2–3
   per-frame channels (total residual flow, Kalman speed, `scb.read`).
3. `agglomerative`: option 1 in both spaces, plus the window-level chaos and fragmentation. HDBSCAN
   as a variant if option 1 shows signal.
4. `prototype`: option 4, fitted per fold.
5. `sync + best clustering`: whether clustering adds to option 0 rather than replacing it.

**The test of the claim:**
- Held-out MAE and RMSE (training scores alongside), against `null` and against `sync`.
- Clustering is justified only if (3), (4) or (5) beat (2) by more than the spread across folds.
- **If none of them beats `null`,** class-relative information is not visible in these features.
  That bears on FYP-1's first claim.
- The mean-of-experts benchmark paragraph is reported as in every experiment from 08 on.

**NaN rules:**
- n = 0: every key NaN.
- n < 3: per-student deviation and synchrony are NaN; clustering features are NaN for n < 4.
- Fewer than 8 frames shared with the rest of the class: that student's synchrony keys are NaN.

Our windows have n ≥ 7 whenever any student is visible, so in practice these rules only apply to
empty windows and to occlusion.

## 15.7 How it plugs into the code

1. **The CFMU contract** (`src/cfmu/extractor.py`). Today `window_features` returns one dict that
   joins Vec, so every student in a window gets the same values.
   - Extend it to return `(window_dict, per_track_dicts)`.
   - `Vec.aggregate_with_tracks` (`src/aggregation/vec.py:56-72`) merges the per-track dicts into
     the student rows as `trk.cluster.*` keys, and the window dict into Vec as `cluster.*`.
   - The `null` extractor returns `({}, [{}] * n)`.
   - Guide 6 §6.6 and recipe §10.8 change with it.
2. **Per-window extractors** (`sync`, `agglomerative`) live in `src/cfmu/` and register with
   `@CLUSTER_EXTRACTORS.register(...)`.
   - They receive the window's sliced tracklets, whose `signals` hold the per-frame channels, and
     the per-track BFMU features.
   - Per-track standardisation (R6) needs session-level statistics per track. The extractor
     computes them once per session from the full tracklets, keyed by track id.
3. **The prototype codebook** is fitted per fold. It is computed in `scripts/train_models.py` from
   the cached `trk.*` rows of each window, grouped by window centre, not in the feature pass.
   - The same code runs at inference, using the final model's codebook.
4. **Caching.** A new CFMU changes `feature_hash` but not `tracking_hash` (`src/core/pipeline.py:46-56`).
   - `python -m scripts.run_pipeline extract --reuse` (or `parallel_runs extract`) rebuilds only
     the feature pass from the cached tracks: minutes per session, not the frame pass's 40.
   - Augmented copies are included.
5. **Tests**, in the style of §10.3 (small synthetic inputs, known answers):
   - every key present with n = 0, 1, 2, 20;
   - permutation invariance;
   - a student whose series is the class median shifted by k frames gets lag k;
   - two well-separated groups of students form two clusters, and fragmentation is near 1;
   - identical students give chaos 0;
   - the per-track keys reach the student rows.

## 15.9 Finding the features: the screening scripts

Before the extractor contract changes (§15.7), the clustering experiment runs **offline**, from the
caches `run_pipeline extract` writes. Scripts in `scripts/clustering/`, maths in
`src/cfmu/classrel.py` (tested in `tests/test_classrel.py`). The extractors of §15.7 will call the
same functions, so whatever the screens pick is computed identically in the pipeline.

The work splits into two questions: **A**, what to cluster on (which signals, which space), and
**B**, which features made from the clusters help. Run them in this order: each screen is cheaper
than the next and removes candidates before it.

| # | Script | Question | Labels? | Output (in `outputs/experiments/10_clustering/`) |
|---|---|---|---|---|
| A1–A3 | `signal_screen` | Is the channel above the tracker's jitter (noise ratio, lag-1 autocorrelation)? How much of each descriptor's variance is within a track, where the labels move? Does the class move together more than when each student's series is circularly shifted? | no | `signal_screen_channels.csv`, `signal_screen_descriptors.csv`, `signal_screen.png` |
| A4 | `event_check` | Do the students move together at moments when the class should react? Events come from half of the class and are tested on the other half, or are given with `--events` | no | `event_check.csv`, `event_times.csv`, `event_check_<channel>.png` |
| A5 | `descriptor_subset` | Which descriptors to cluster on: drop seat-like ones (A2), rank by Laplacian score, keep one per redundancy group | no | `descriptor_subset.csv`, `descriptor_subset.json` (the set `subset` the later scripts read) |
| A6 | `stability` | How stable is each clustering under subsampling of the students, and between overlapping windows? Which thresholds τ are degenerate? | no | `stability.csv`, `stability.png` |
| B7 | `candidates` | Every candidate of options 0, 1 and 2 on the windows centred on rating entries, at the τ chosen in A6 | no | `candidates/<feature hash>-<settings hash>/<stem>.json` (`--format csv` for flat tables) |
| B8 | `feature_screen` | Within-student Spearman ρ and mutual information with the ratings, on the **training** groups of each fold (the held-out ρ is shown, not used) | train folds | `feature_screen.csv`, `feature_screen_folds.csv`, `feature_screen.png` |
| B9 | `importance` | Held-out permutation importance of each feature family in the configured per-labeler model; `--gate` also tests the window-level features in the activity gate | held-out | `importance*.csv`, `importance.png` |
| B10 | `ablation` | The arms of §15.6 (`null`, `sync`, `agglomerative`, `hdbscan`, `prototype`, `sync+best`) on train_models' folds and metrics, with the §15.6 test | held-out | `ablation/<config hash>-<settings hash>.json`, `ablation.csv`, `ablation_verdict.csv` |
| – | `report` | Every result above gathered in one file, to carry back from the machine that ran them | – | `clustering_report.json` |

```
python -m scripts.clustering.signal_screen
python -m scripts.clustering.event_check
python -m scripts.clustering.descriptor_subset
python -m scripts.clustering.stability
python -m scripts.clustering.candidates --tau-desc <from stability> --tau-sync <from stability>
python -m scripts.clustering.feature_screen
python -m scripts.clustering.importance --gate
python -m scripts.clustering.ablation --best <agglomerative|hdbscan|prototype>
python -m scripts.clustering.report
```

Every script takes `--config` (default `docs/experiments/09_vfoa.json`, whose head-pose channels
are the natural input), `--set`, `--sessions` and `--out`. Every result table is written as `.csv`
and `.json`. `signal_screen`, `stability` and `candidates` take `--workers N` (sessions in parallel,
CPU). The candidates are stored as one JSON file per session (`--format csv` or `both` for flat
tables). The full-dataset run, from a fresh GPU machine to the report, is in
[`scripts/clustering/README.md`](../../scripts/clustering/README.md).

**What needs what:**
- A1–A6 and B7 run on any cached session, including a single one.
- B8–B10 need label rows from at least two groups, because they use leave-one-group-out folds.
- B10 also needs the augmented copies the config names (`augmentation.copies`), as `train_models` does.

**Choices made here** (the screens report them, so they can be revisited):
- Per-track standardisation uses the regular-grid rows of the session (`tracks.csv`). Where a track
  has fewer than 5 windows, or a flat descriptor, it uses the whole session's statistics instead.
- A noise ratio of 1.5 is the default pass mark (`--noise-pass`).
- Synchrony "passes" with a one-sided sign test p < 0.05 over windows that the real correlation
  exceeds the surrogate mean.
- An event-response "move" is a frame-to-frame change above the student's own 90th percentile.
- The prototype codebook is fitted per fold on the training groups' label-window descriptors, with
  augmented copies included and no labels (option 4, way 2).
- `sync+best` chosen automatically uses the held-out folds. Freeze `--best` before a result is cited.
- Window-level `cluster.*` columns are included by default. They are constant within a window and
  can identify sessions, as the class context did in experiment 08, so `--no-window` is the check.

## 15.8 References

All were checked against the publisher or an index record on 17 November 2026.

- Raca, M., Tormey, R. and Dillenbourg, P. Sleepers' lag: study on motion and attention. *LAK 2014*;
  *Journal of Learning Analytics* 3(2), 2016, doi:10.18608/jla.2016.32.12.
- Goldberg, P., Sümer, Ö., Stürmer, K., Wagner, W., Göllner, R., Gerjets, P., Kasneci, E. and
  Trautwein, U. Attentive or not? Toward a machine learning approach to assessing students' visible
  engagement in classroom instruction. *Educational Psychology Review*, 2019,
  doi:10.1007/s10648-019-09514-z.
- Dikker, S. et al. Brain-to-brain synchrony tracks real-world dynamic group interactions in the
  classroom. *Current Biology* 27(9), 1375–1380, 2017, doi:10.1016/j.cub.2017.04.002.
- Ester, M., Kriegel, H.-P., Sander, J. and Xu, X. A density-based algorithm for discovering
  clusters in large spatial databases with noise. *KDD 1996*, 226–231.
- Campello, R. J. G. B., Moulavi, D. and Sander, J. Density-based clustering based on hierarchical
  density estimates. *PAKDD 2013*, LNCS 7819, 160–172, doi:10.1007/978-3-642-37456-2_14.
- Chakrabarti, D., Kumar, R. and Tomkins, A. Evolutionary clustering. *KDD 2006*, 554–560,
  doi:10.1145/1150402.1150467.
- Spiliopoulou, M., Ntoutsi, I., Theodoridis, Y. and Schult, R. MONIC: modeling and monitoring
  cluster transitions. *KDD 2006*, 706–711, doi:10.1145/1150402.1150491.
- Blondel, V. D., Guillaume, J.-L., Lambiotte, R. and Lefebvre, E. Fast unfolding of communities in
  large networks. *J. Stat. Mech.* 2008, P10008, doi:10.1088/1742-5468/2008/10/P10008.
- Traag, V. A., Waltman, L. and van Eck, N. J. From Louvain to Leiden: guaranteeing well-connected
  communities. *Scientific Reports* 9, 5233, 2019, doi:10.1038/s41598-019-41695-z.
