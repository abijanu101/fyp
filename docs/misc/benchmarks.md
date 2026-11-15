# DIPSER: published benchmarks

What others have reported on DIPSER, how they did it, and how far their numbers can be compared
with ours. Literature searched in November 2026 (arXiv, Semantic Scholar citations, publisher
pages, GitHub). Anything that could not be read in full is marked.

**In short:** four papers and one informal project report numbers on DIPSER. All of them use the
**per-student desk cameras** (face or upper body of one student), most with smartwatch data; none
uses the context (classroom) cameras, multi-person tracking, or a held-out group of students. **No
published number is directly comparable with this project's setting**, which is therefore a new
protocol, not a comparison against a state of the art. Measured against what a constant
prediction of "3" scores on the same metric (§1), two of the four prediction results are worse
than the constant, one beats it by about 9%, and one beats it clearly on a filtered target (§3).

Contents: §1 trivial baselines (the floor everything is read against) · §2 how the numbers are
computed, and why the same metric can mean different things · §3 published works, worst to best ·
§4 the dataset paper · §5 what this means for this project.

## 1. Trivial baselines: predicting 3 every time

None of the published works reports a constant-prediction baseline for every metric it uses, so
we computed them (`python -m scripts.label_baselines`): all 483 labelled student-sessions, scored
against each target used in §3.

**The headline numbers** (per second, pooled; §2 explains both choices):

| Target | Used by | RMSE | MAE | Within ±1 | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|---|---|---|---|
| Mean of the expert labelers (regression) | §3.3 | **0.440** | **0.337** | – | – | – | – |
| Majority vote, all six labelers incl. self | §3.1 | 0.638 | 0.388 | 0.990 | 0.622 | **0.477** | 0.153 |
| Majority vote, experts only | §3.4 | 0.645 | 0.389 | 0.987 | 0.624 | 0.480 | 0.154 |
| Each labeler's own ratings, pooled | – | 0.819 | 0.586 | 0.958 | 0.457 | 0.286 | 0.125 |

Predicting the target's training mean instead of 3 on the averaged-expert target gives RMSE 0.438
and MAE 0.341: on that target "predict 3" and "predict the mean" are the same baseline.

**Binary** (not engaged = attention below 3, as in §3.4). A constant 3 counts as "engaged": its
not-engaged F1 is **0**, at 79.1% accuracy against the expert majority (75.5% against the
six-labeler majority). The opposite constant, always "not engaged", scores not-engaged F1 0.35
(expert majority) to 0.48 (individual labelers), because F1 on a minority class rewards
predicting it everywhere. A random scorer has AUC 0.5.

**Why a constant does so well.** The majority vote is 3 for 62% of seconds, 2 for 25%, 4 for 12%,
1 or 5 almost never; a constant 3 is within one point of it 99% of the time. The averaged expert
label has a standard deviation of only 0.44.

**Sanity check against the papers.** §3.3 reports MAE 0.312 for predicting the training mean on its
9 test subjects; over the whole dataset we get 0.341. §3.4 reports 78.8% accuracy for always
predicting the majority; we get 79.1%.

**The same constant, computed four ways.** RMSE / MAE / accuracy / weighted F1 / macro F1 of
"predict 3" (accuracy and F1 only for the vote targets):

| Target | Per second, pooled | Per second, per subject | Per change, pooled | Per change, per subject |
|---|---|---|---|---|
| Mean of experts (RMSE / MAE) | 0.440 / 0.337 | 0.402 / 0.336 | 0.464 / 0.364 | 0.437 / 0.361 |
| Majority, six labelers (RMSE / MAE) | 0.638 / 0.388 | 0.576 / 0.389 | 0.785 / 0.563 | 0.748 / 0.539 |
| Majority, six labelers (acc / wF1 / mF1) | 0.622 / 0.477 / 0.153 | 0.621 / 0.524 / 0.144 | 0.464 / 0.294 / 0.127 | 0.483 / 0.320 / 0.129 |
| Majority, experts (acc / wF1 / mF1) | 0.624 / 0.480 / 0.154 | 0.624 / 0.530 / 0.144 | 0.450 / 0.280 / 0.124 | 0.485 / 0.326 / 0.129 |
| Majority, experts: engaged accuracy | 0.791 | 0.791 | 0.642 | 0.645 |
| Majority, experts: not-engaged F1 of "always not engaged" | 0.346 | 0.290 | 0.527 | 0.502 |
| Each labeler (RMSE / MAE) | 0.819 / 0.586 | 0.797 / 0.586 | 0.892 / 0.663 | 0.863 / 0.648 |
| Samples | 140,572 s (702,265 labeler-seconds) | 483 subjects | 30,600 / 13,552 / 14,660 / 44,103 runs | 483 subjects |

The constant's accuracy moves from 0.62 to 0.45 and its RMSE from 0.64 to 0.80 on the *same labels*
just by changing what counts as one sample. A published number without its sampling and pooling
stated has an uncertainty of that size before any model is involved.

**One more reference: each student's own average.** Per subject, predicting that student's mean
attention gives MAE 0.192 (per second) on the mean-of-experts target, against 0.336 for the
constant 3. Most of the label variance is *between* students, not within a session: a model that
only learned what each student is like on average would beat every published regression number.
It cannot be used under a by-subject split (the test students are unseen), but it is why a split
that lets the same student appear in training and test (§3.1's same-student few-shot) inflates
results.

## 2. How the numbers are computed

### 2.1 The metrics

- **MAE** (mean absolute error): the average distance between predicted and true score, in
  attention points. **RMSE** (root mean squared error): the same with squared distances, so large
  misses count more. Both: lower is better, on the 1–5 scale.
- **Within ±1 ("one-off") accuracy**: the share of predictions no more than one point from the
  label.
- **Accuracy**: share of exact matches. **Balanced accuracy**: the mean of per-class recalls, so
  every class counts equally (chance for five classes is 0.20).
- **F1** of one class: the harmonic mean of precision (of the predictions of that class, how many
  were right) and recall (of the true cases of that class, how many were found). **Macro F1**:
  the plain mean over classes. **Weighted F1**: the mean weighted by how often each class occurs,
  so the dominant class decides most of it. **Not-engaged F1**: F1 of the minority class alone.
- **AUC** (area under the ROC curve): the probability that a randomly chosen positive gets a higher
  score than a randomly chosen negative. 0.5 is chance; it is independent of any threshold.
- **Pearson r**: linear correlation between predicted and true values; indifferent to offset and
  scale, so a constant scores nothing (r undefined, treated as 0).
- **ICC** (intraclass correlation): agreement between raters on the same items, from 0 (none) to 1.

### 2.2 Choices that change the number, often unstated

The first two (target and unit) are laid out as a full grid, with the constant baseline of every
cell and where each work sits, in `docs/misc/targets.md`.


1. **The target.** Mean of experts, majority vote (with or without the self-report), or each
   labeler's own rating; for votes, how ties are broken or whether tied seconds are dropped; for
   binary, where the threshold sits (§3.4 uses 1–2 vs 3–5). Same students, very different floors
   (§1 table).
2. **Per second vs per change.** DIPSER labels are change points: a labeler records a value when
   it changes, and it holds until the next one. Sampling the held value every second (or every
   frame) counts a ten-minute stretch of "3" 600 times; counting each change once counts it once.
   Per second favours whatever is common and long-lasting (the constant 3); per change gives every
   labelled episode equal weight, including short dips. Frame-level and window-level evaluation
   (§3.3's 10-second windows, §3.4's 10-frame windows) behave like per second. Sampling one image
   every 2 s (§3.1) is per second at half the rate.
3. **Pooled vs per subject.** Pooling puts every sample in one pile, so long sessions dominate;
   averaging per subject (or per session) gives every student equal weight. Per subject lowers
   RMSE (each student's errors are measured against a narrower spread) and, for classes,
   raises weighted F1.
4. **Filtering.** Dropping frames without a face (§3.3), seconds where the vote is tied, or windows
   whose frames carry different labels (§3.4) removes exactly the ambiguous moments and makes the
   remaining task easier. The constants above use every second.
5. **The split.** Random or same-student splits let the model recognise the student (§1, last
   paragraph); by-subject splits do not; by-group splits (ours) also hold out the room, days and
   cohort.
6. **Overlap.** Overlapping windows share frames, so test samples are not independent and the
   spread across seeds understates the real uncertainty.

## 3. Published works, worst to best

Ranked by how the reported number compares with the constant of §1 *on the same metric and target
as far as it is known*. The comparison is approximate where the protocol is not fully stated.

| # | Work | Input | Target | Split | Reported | Trivial constant on that metric | Verdict |
|---|---|---|---|---|---|---|---|
| 3.1 | Marquez-Carpintero, Viejo, Cazorla, *IEEE Access* 2025 | desk camera, activities 8–9 | majority vote, 6 labelers | few-shot, same vs other students | weighted accuracy **0.116** across students (0.413 same student) | 0.622 accuracy (0.20 if balanced) | below the constant |
| 3.2 | Liu et al., *ISET* 2025 | face video + wearables | "attention", unknown | unknown | RMSE **0.87**, within ±1 76.3% | RMSE 0.64–0.80, within ±1 0.97–0.99 on any vote target; 0.82–0.89 / 0.93–0.96 on individual ratings | below the constant |
| 3.3 | Fragkiadakis et al., arXiv 2026 | desk-camera frames (CLIP) + smartwatch | mean of experts | by subject 39/9/9 | MAE **0.283**, RMSE 0.363 | MAE 0.312, RMSE 0.393 (their train mean) | 9% better |
| 3.4 | GitHub `shgfyyy-7029/classroom-engagement-recognition` | DIPSER's face-mesh and pose features | binary majority, ties and mixed windows dropped | by subject 38/8/11 | not-engaged F1 **0.524**, AUC 0.786 | F1 0 (always engaged) / ~0.35 (always not engaged); AUC 0.5 | clearly better, on a filtered target |
| 3.5 | Arranz-Romero et al., *Applied Sciences* 2026 | emotion + attention index | correlation with an engagement criterion | by subject | r = 0.166 | r = 0 | not a prediction benchmark |

Works that cite DIPSER without reporting results on it: Lu et al. (*Scientific Data* 2025), Tran et
al. (WACV 2026), Petković et al. (VISAPP 2026), Babanne et al. (I2ITCON 2026), Moore et al. (IMWUT
2025), an ICEMCSI 2026 paper. Semantic Scholar lists 9 citing works and Google Scholar 15; the
remaining few could not be enumerated, so a result may have been missed.

### 3.1 IEEE Access 2025: multimodal language models (the dataset authors). Below the constant.

*Enhancing Engineering and STEM Education With Vision and Multimodal Large Language Models.*
DOI 10.1109/ACCESS.2025.3584025. Read in full during the first literature search (a second fetch
for details was refused by the publisher, so prompt formats below are as first summarised).

- **Data:** activities 8 (robotics) and 9 (activity design) only, 57 students, 14,425 desk-camera
  images (one every 2 s).
- **Target:** majority vote of six annotators (five experts plus self) per second, held until the
  majority changes; class 3 dominates.
- **Method.** LLaVA v1.6-34B is a *multimodal large language model*: a 34-billion-parameter
  language model that also reads images. It is not trained on DIPSER; it is *prompted*: given an
  image and a text question ("rate this student's attention from 1 to 5") and its text answer is
  read as the score.
  - *Zero-shot*: the image and the question only.
  - *Few-shot* (8 or 16 shots): the prompt first shows 8 or 16 example images with their correct
    labels, then the test image. The examples come either from **the same student and activity**
    (so the model sees the student it is judging, with labels) or from **other students**.
  - Variants: rescaled label scales, face crops instead of the full frame, a history of previous
    frames, and smartwatch values written into the prompt as text.
  - A small CNN (convolutional image classifier) trained on the images, for comparison.
- **Labeler agreement** (each labeler against the majority vote): weighted precision 0.405–0.611,
  weighted F1 0.414–0.610, MAE 0.53–0.86. The self-report agrees best (F1 0.610), the worst expert
  at 0.414. The authors call it "around 0.6". Each labeler is part of the majority it is compared
  with, which flatters these numbers.
- **Results** (attention, "weighted accuracy"): 8-shot same student 0.413; 16-shot same student
  0.386; **16-shot other students 0.116**; best zero-shot 0.301; CNN 0.325 (split unclear).
  Smartwatch data gave no consistent gain.
- **Against the floor:** "weighted accuracy" is not defined. If it is support-weighted recall it
  equals plain accuracy, where the constant scores 0.62; if it is balanced accuracy, chance is
  0.20. Either way the honest cross-student number (0.116) is below a constant, and even the
  same-student number (which uses labelled examples of the test student) is below 0.62 if the
  metric is plain accuracy. Note also that a constant's weighted F1 (0.48) is above the worst
  labeler's agreement (0.414): weighted F1 against a majority rewards the dominant class.

### 3.2 Liu et al., ISET 2025: face and wearables. Below the constant.

*Individual student attention detection in face-to-face classrooms* (index:
<https://scholars.hkmu.edu.hk/en/publications/individual-student-attention-detection-in-face-to-face-classrooms/>).
Only the abstract was readable (paywalled).

- **Method (from the abstract):** a fusion network that combines facial video with wearable-sensor
  data, reported better than either alone.
- **Results:** RMSE 0.87, one-off (within ±1) accuracy 76.3%.
- **Against the floor:** on any consensus target the constant 3 has RMSE 0.64–0.80 and is within
  one point 97–99% of the time; even on individual labelers' ratings it reaches RMSE 0.82–0.89 and
  within ±1 93–96%. Unless the target is something else entirely (a different scale or a
  continuous index), the model is worse than the constant on both of its metrics. Not comparable
  until the protocol is known.

### 3.3 Fragkiadakis et al., arXiv 2026: fairness-aware attention regression. 9% better than the mean.

*Fairness-Aware Multimodal Transformer Modeling for Real-Time Student Attention Estimation.*
arXiv:2609.02232 (<https://arxiv.org/html/2609.02232>). Read in full; the closest to our label
handling. Some architecture sizes come from appendix tables that render poorly in HTML.

- **Target:** the **mean of the expert labelers** (self-reports excluded), per second, as a 1–5
  regression. Each sample is a **10-second window** of 10 one-second steps; the target is the
  label at the **last** second, the earlier nine are context.
- **Split:** 57 subjects by subject, 39 / 9 / 9 (75,189 / 17,208 / 15,636 windows), stratified by
  age group, at least 3 male subjects in each held-out set; 10 training seeds, plus 10 repeated
  subject splits.
- **Inputs:**
  - *Visual:* for each second, the first valid desk-camera frame, resized and centre-cropped to
    224×224 and passed through **CLIP ViT-L/14**, a large pretrained image model, used as a fixed
    feature extractor. Each frame becomes a 768-number *embedding* (a summary of the image's
    content learned from image-caption pairs), plus a flag for a missing frame. Frames without a
    visible face are dropped.
  - *Sensors:* smartwatch heart rate, accelerometer and gyroscope, averaged over each second (mean
    per axis and mean magnitude, heart rate below 30 bpm discarded).
- **Models.** A **GRU** (gated recurrent unit) is a small recurrent network: it reads a sequence
  one step at a time and keeps a running memory, so its output after the tenth second depends on
  all ten.
  - *Visual GRU*: CLIP embeddings → linear layer → one GRU (hidden size 128) → small regression
    head → one score.
  - *Sensor GRU*: the same idea on smartwatch data only: separate GRUs for motion (hidden 64) and
    heart rate (hidden 16), concatenated, then a regression head. No camera input.
  - *Residual fusion transformer*: the visual stream (a causal transformer over the CLIP
    embeddings) makes the main prediction; separate small transformers summarise motion and heart
    rate; a *gate* (a learned number between 0 and 1) decides how much of a sensor-based
    *correction* (capped at ±0.5) is added to the visual prediction. "Visual primary, gated sensor
    correction."
  - *Inverse-frequency-weighted MSE*: labels are grouped into bins and each training sample's
    squared error is weighted by (N / (K · n_bin))^0.5, so rare attention levels count more and the
    model is pushed away from always predicting the common value. The bins are not defined in the
    paper.
- **How "fairness-aware" works.** Gender (female/male) and age (four bins) are *estimated* from the
  faces by DIPSER's precomputed age/gender models (MiVOLO, DeepFace), not self-reported. During
  training, each batch is built with equal numbers of subjects per group (2 per gender, or 1 per
  age bin, 8 windows each), and a penalty is added to the loss: the **largest group's MAE minus the
  smallest group's MAE** (per-subject MAE averaged within each group), times a weight λ (0.7 for
  gender, 0.8 for age, chosen on validation data). It is the only fairness method used; there is
  no adversarial or post-hoc correction.
- **Results** (test, mean ± SD over 10 seeds):

  | Model | MAE | RMSE |
  |---|---|---|
  | Predict the training mean | 0.312 | 0.393 |
  | Sensor GRU | 0.313 ± 0.002 | 0.398 ± 0.003 |
  | Visual GRU | 0.287 ± 0.007 | 0.366 ± 0.008 |
  | Residual fusion transformer | 0.283 ± 0.009 | 0.363 ± 0.009 |

  The sensor-only model is no better than the mean. 1 fps and 10 fps give the same result (5 seeds).
  The fairness penalty narrowed the validation gaps but **not the test gaps** (gender gap 0.015 →
  0.022, age gap 0.126 → 0.150; reduced in only 2–4 of 10 runs): with 9 test subjects, 3 of them
  male and one in the oldest age bin, the group estimates are unstable. Ordinal classification and
  early/late fusion were tried and did not help.
- **Against the floor:** the best model improves on predicting the mean by about 0.03 MAE (9%)
  with a CLIP-L backbone on close-up faces. Pooled or per-subject averaging of the reported MAE is
  not stated; §1 shows that choice alone moves the constant's RMSE from 0.44 to 0.40.

### 3.4 Informal GitHub project. Clearly better, on a filtered target.

<https://github.com/shgfyyy-7029/classroom-engagement-recognition>. Not peer-reviewed; README in
Chinese; code read.

- **Features:** DIPSER's precomputed per-frame features, 1,544 numbers: MediaPipe face mesh (478
  landmarks × 3 = 1,434), body pose (33 × 3 = 99), head pose (pitch, yaw, roll), face and body boxes
  (4 + 4). Missing values filled with 0; not standardised. Camera and frame rate not stated.
- **Target:** four labelers (`labeler_01`–`04`), each forward-filled to every frame, mapped to low
  (1–2) / mid (3) / high (4–5); the frame label is their majority vote, **frames with a tied vote are
  dropped**. Binary: low = not engaged, mid or high = engaged.
- **Windows:** 10 consecutive frames, stride 5 (50% overlap). **A window is kept only if all ten
  frames have the same label**, so transitions are dropped.
- **Model:** a 2-layer GRU (hidden 128) over the 10 frames, last hidden state → small classifier;
  not-engaged windows oversampled 2× during training; the decision threshold chosen on validation
  data to maximise not-engaged F1. Logistic regression and an MLP as comparisons.
- **Split and metric:** by subject, stratified by group, 38 / 8 / 11 (30,412 test windows, 79%
  engaged); metrics pooled over all test windows; 3 seeds.
- **Results:** not-engaged F1 **0.524**, accuracy 74.9%, AUC 0.786; logistic regression F1 0.373,
  MLP 0.408. Head pose alone (3 numbers) F1 0.447, body pose alone 0.476: pose carries most of the
  signal. Adding smartwatch data lowered F1 (0.27 vs 0.50). 3-class accuracy 52.5% (exploratory);
  1–5 regression r 0.21, called unusable by the author.
- **Against the floor:** always "engaged" gives F1 0 and 78.8% accuracy; always "not engaged"
  gives F1 about 0.35 on a 21% minority; AUC 0.786 is well above 0.5. The clearest gain of any
  work, but on a target with ties and transitions removed (§2.2 point 4), with overlapping windows
  and 11 test subjects.

### 3.5 IPA 2.0, Applied Sciences 2026. Not a prediction benchmark.

DOI 10.3390/app16052515. Abstract only (publisher page refused access). Validates an interpretable
attention index rather than predicting labels: r = 0.166 (95% CI 0.017–0.308) over 172 evaluable
units, with subject-wise calibration and an explicit anti-leakage audit.

## 4. The dataset paper

Marquez-Carpintero, Suescun-Ferrandiz, Lorenzo Álvarez, Fernandez-Herrero, Viejo, Roig-Vila,
Cazorla (University of Alicante). *DIPSER: A Dataset for In-Person Student Engagement Recognition in
the Wild.* arXiv:2502.20209, 2025. <https://arxiv.org/abs/2502.20209>. Also released as *DIPSEER*
(Science Data Bank, DOI 10.57760/SCIENCEDB.11541), the folder name our copy uses. No peer-reviewed
version found. Code: <https://bitbucket.org/rovitlib/dipser/>.

- **Contents:** 3 groups (two undergraduate, one master's), 9 activities, about 51 hours; 5–6
  context cameras at 1280×720, a desk camera per student at 640×480, a smartwatch per student
  (heart rate, acceleration, gyroscope, light). Precomputed face and body boxes, head pose,
  MediaPipe face mesh and pose, and DeepFace attributes are distributed with it.
- **Labels:** attention 1–5 and nine emotion classes, as change-point annotations at one-second
  resolution (annotators saw one frame per second), by expert labelers plus the student's
  self-report.
- **Baselines: none.** The paper reports no models, splits or metrics; validation is qualitative
  (sensor plots, a t-SNE separating two activities, clock synchronisation within 0.5 s).
- **Labeler agreement: not reported** in the dataset paper (see §3.1 for the authors' later
  figures).

**Discrepancies with our copy and between sources:**

| Item | Sources say | Our copy |
|---|---|---|
| Context-camera frame rate | about 9 fps (paper) | **5.00 fps** measured on every extracted session (`timing_audit`); likely a property of the distributed export, unconfirmed |
| Expert labelers | 4 experts + self (arXiv); 5 experts + self (IEEE Access) | 4 experts + self per student, from a pool of 5: `labeler_01`–`03` always, plus `labeler_04` (178 student-sessions) or `labeler_05` (143) |
| Subjects | 54 (arXiv Table 1); 57 (both follow-ups) | 483 labelled student-sessions |
| Licence | CC BY with regulated access (arXiv); CC BY-NC 4.0 (Science Data Bank) | check before redistribution of any derived data |

## 5. What this means for this project

**How our setting differs from every published one:**

| | Published works | This project |
|---|---|---|
| Camera | per-student desk camera, face close-ups | classroom context cameras, many students per frame |
| People | one known student per stream | multi-person detection and tracking, identities unknown at inference |
| Unit predicted | one student | the class per window (a retention curve), via per-student scores in late aggregation |
| Split | random, same-student few-shot, or by subject | **leave one group out**: unseen students, room setup and days |
| Labels | majority vote, or mean of experts | median of labelers (self included by default), thresholded, then share of engaged students |
| Models | LLaVA-34B, CLIP-L + GRU, CNNs | classical ML on hand-built features |

**Lessons that transfer:**

1. **The task is hard everywhere.** Labelers agree with their own majority at F1 0.41–0.61; the
   best close-up face model beats predicting the mean by 9% in MAE; across students, a 34B
   multimodal model scores below a constant. Our near-chance six-session result
   (`docs/experiments/05_six_session_run.md`) is in line with this, not an outlier.
2. **Always report the trivial baseline**, computed on the same target, sampling and pooling (§1).
   We already report `majority`; for a continuous target (direction 3 of `05_six_session_run.md`
   §7) add a predict-the-mean MAE. Prefer metrics a constant cannot game: macro F1, minority-class
   F1, AUC or correlation over weighted F1 and accuracy.
3. **State how the metric was computed**: target, per second or per change, pooled or per subject,
   what was filtered (§2.2). Report per second *and* per change where it matters: they differ by
   as much as a model's whole margin over the floor.
4. **Report against human agreement.** Labeler-vs-consensus agreement gives a ceiling; Petković et
   al. (VISAPP 2026, on their own data) report their model (r = 0.53) next to inter-rater ICC 0.85,
   a good template.
5. **Averaged labels make a low-variance target**, which shrinks MAE without the model learning
   much: predicting 3 already gives RMSE 0.44 and MAE 0.34 on the mean of the experts. Prefer
   correlation (Pearson or Spearman; a within-session curve correlation is proposed in
   `docs/misc/proposed_metrics.md`) alongside MAE.
6. **Most label variance is between students** (§1, last paragraph). Within-session tracking of
   change, which our retention curve is about, is the harder and more honest question.
7. **Pose and head orientation carry most of the signal** where measured (§3.4). Both can be
   estimated from context cameras; head pose is a candidate feature family for us.
8. **Smartwatch data added nothing** in three independent works; not using it costs little.
9. **Frame rate matters little:** 1 fps equalled 10 fps for prediction (§3.3), consistent with
   labels annotated at one frame per second.

**How to present our results:** as the first context-camera, group-held-out benchmark on DIPSER,
with its own baselines (majority, predict-the-mean, hand-weighted) and human agreement as reference,
and an explicit note that per-student desk-camera numbers measure a different, easier task.

## Not verified

The full text of Liu et al. (paywalled) and of IPA 2.0 (access refused); details of the IEEE Access
prompts on a second reading (publisher refused the fetch); about six works that Google Scholar counts
as citing DIPSER but Semantic Scholar does not list. In §3.4, three result files (3-class,
regression, smartwatch fusion) were summarised from the README, not read.
