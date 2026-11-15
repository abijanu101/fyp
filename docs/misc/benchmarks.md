# DIPSER: published benchmarks

What others have reported on DIPSER, how they did it, and how far their numbers can be compared
with ours. Literature searched in November 2026 (arXiv, Semantic Scholar citations, publisher
pages, GitHub). Anything that could not be read in full is marked.

**In short:** four papers and one informal project report numbers on DIPSER. All of them use the
**per-student desk cameras** (face or upper body of one student), most with smartwatch data; none
uses the context (classroom) cameras, multi-person tracking, or a held-out group of students. **No
published number is directly comparable with this project's setting**, which is therefore a new
protocol, not a comparison against a state of the art. The most useful reference point is a
regression result on labeler-averaged attention that barely beats predicting the mean (§2.3).

## 1. The dataset paper

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
- **Labeler agreement: not reported** in the dataset paper (see §2.1 for the authors' later
  figures).

**Discrepancies with our copy and between sources:**

| Item | Sources say | Our copy |
|---|---|---|
| Context-camera frame rate | about 9 fps (paper) | **5.00 fps** measured on every extracted session (`timing_audit`); likely a property of the distributed export, unconfirmed |
| Expert labelers | 4 experts + self (arXiv); 5 experts + self (IEEE Access) | 4 experts + self per student, from a pool of 5: `labeler_01`–`03` always, plus `labeler_04` (178 student-sessions) or `labeler_05` (143) |
| Subjects | 54 (arXiv Table 1); 57 (both follow-ups) | not recounted here |
| Licence | CC BY with regulated access (arXiv); CC BY-NC 4.0 (Science Data Bank) | check before redistribution of any derived data |

## 2. Works reporting results

| # | Work | Input | Target | Split | Headline result |
|---|---|---|---|---|---|
| 1 | Marquez-Carpintero, Viejo, Cazorla, *IEEE Access* 13, 2025 (dataset authors) | desk camera, activities 8–9 only; some smartwatch | attention 1–5, majority vote of labelers | few-shot from the same student vs other students | weighted accuracy 0.413 (same student) / **0.116 (other students)** |
| 2 | Liu, Chui, Lee, Ng, Paoprasert, Zhao, *ISET 2025* | facial video + wearables | continuous attention | unknown (paywalled) | RMSE 0.87, one-off accuracy 76.3% |
| 3 | Fragkiadakis, Ziabari, Alsahag, arXiv:2609.02232, 2026 | face crops from the desk camera (CLIP embeddings) + smartwatch | mean of expert labelers (no self-report), regression | by subject, 39/9/9 | MAE 0.283 vs **0.312 for predicting the training mean** |
| 4 | Arranz-Romero, Roig-Vila, Cazorla, *Applied Sciences* 16(5), 2026 | emotion + attention index | correlation with an engagement criterion | by subject, hold-out | r = 0.166 [0.017, 0.308] |
| 5 | GitHub `shgfyyy-7029/classroom-engagement-recognition` (informal) | DIPSER's precomputed face-mesh and pose features | engaged / not (majority vote) | by subject, 38/8/11 | not-engaged F1 0.524, AUC 0.786 |

Works that cite DIPSER without reporting results on it: Lu et al. (*Scientific Data* 2025), Tran et
al. (WACV 2026), Petković et al. (VISAPP 2026), Babanne et al. (I2ITCON 2026), Moore et al. (IMWUT
2025), an ICEMCSI 2026 paper. Semantic Scholar lists 9 citing works and Google Scholar 15; the
remaining few could not be enumerated, so a result may have been missed.

### 2.1 IEEE Access 2025: multimodal language models (the dataset authors)

*Enhancing Engineering and STEM Education With Vision and Multimodal Large Language Models.*
DOI 10.1109/ACCESS.2025.3584025. Read in full.

- **Data:** activities 8 (robotics) and 9 (activity design) only, 57 students, 14,425 desk-camera
  images (one every 2 s).
- **Labels:** majority vote of six annotators (five experts plus self) per second, held until the
  majority changes; class 3 dominates.
- **Method:** LLaVA v1.6-34B prompted zero-shot and few-shot (variants: rescaled scales, face
  crops, frame history); a small CNN as comparison; smartwatch values in the prompt.
- **Labeler agreement** (each labeler against the majority vote, attention): weighted precision
  0.405–0.611, F1 0.414–0.610, MAE 0.53–0.86. The self-report agrees best (F1 0.610), the worst
  expert at F1 0.414. The authors call it "around 0.6".
- **Results** (attention, weighted accuracy): 8-shot with examples from the *same student and
  activity* 0.413; 16-shot same-student 0.386; **16-shot with examples from other students 0.116**;
  best zero-shot 0.301; CNN 0.325 (split not clearly stated). Smartwatch data gave no consistent gain.
- **Caveats:** the best numbers use examples of the test student (identity leakage by design); the
  honest cross-student number is 0.116, below chance for five classes with a dominant class.

### 2.2 Liu et al., ISET 2025: face and wearables

*Individual student attention detection in face-to-face classrooms* (index:
<https://scholars.hkmu.edu.hk/en/publications/individual-student-attention-detection-in-face-to-face-classrooms/>).
Only the abstract was readable. A fusion network over face video and wearables, reported better
than single modalities: RMSE 0.87, one-off accuracy 76.3%. Split, label processing and camera are
unknown; the RMSE is far from that of §2.3 on a similar scale, which suggests a different target
(majority vote or individual labelers). Not comparable until the protocol is known.

### 2.3 Fragkiadakis et al., arXiv 2026: fairness of attention regression

*arXiv:2609.02232* (<https://arxiv.org/html/2609.02232>). Read in full. The closest to our label
handling.

- **Data:** desk-camera face frames at 1 fps (frames without a face dropped), rolling 10-second
  windows; smartwatch heart rate and motion.
- **Target:** the **mean of the expert labelers**, self-reports excluded, as a 1–5 regression.
- **Split:** by subject, 39 / 9 / 9, stratified by age; 10 seeds and 10 repeated subject splits.
- **Models:** a GRU over CLIP ViT-L/14 embeddings, a sensor GRU, and a fusion transformer
  (visual primary, gated sensor correction); inverse-frequency-weighted MSE.
- **Results** (test, 10 seeds):

  | Model | MAE | RMSE |
  |---|---|---|
  | Predict the training mean | 0.312 | 0.393 |
  | Sensor GRU | 0.313 ± 0.002 | 0.398 ± 0.003 |
  | Visual GRU | 0.287 ± 0.007 | 0.366 ± 0.008 |
  | Fusion | 0.283 ± 0.009 | 0.363 ± 0.009 |

  1 fps and 10 fps give the same result. Fairness regularisers did not reduce gender or age gaps
  reliably (gap reduced in 2–3 of 10 splits).
- **Caveats:** the best model improves on predicting the mean by about 0.03 MAE (9%), with a
  CLIP-L backbone on close-up faces; averaged labels have little variance; 9 test subjects; no
  split by group, so cross-cohort generalisation is untested.

### 2.4 IPA 2.0, Applied Sciences 2026

DOI 10.3390/app16052515. Abstract only (publisher page refused access). Validates an interpretable
attention index rather than predicting labels: r = 0.166 (95% CI 0.017–0.308) over 172 evaluable
units, with subject-wise calibration and an explicit anti-leakage audit. Not a prediction
benchmark.

### 2.5 Informal GitHub project

<https://github.com/shgfyyy-7029/classroom-engagement-recognition>. Not peer-reviewed. Uses
DIPSER's precomputed 1,544-dimensional face-mesh, body-pose, head-pose and box features (desk
camera, by the look of it), 10-frame windows, binary engaged / not engaged by majority vote, split
by subject (38/8/11), test set about 79% engaged.

- GRU on sequences: not-engaged F1 0.524, accuracy 74.9%, AUC 0.786; always predicting the
  majority would score 78.8% accuracy. Logistic regression F1 0.373, MLP 0.408.
- Head pose alone (3 numbers) F1 0.447, body pose alone 0.476: pose carries most of the signal.
- Adding smartwatch data lowered F1 (0.27 vs 0.50). 3-class accuracy 52.5%; 1–5 regression r 0.21,
  called unusable by the author.
- Caveats: overlapping windows, 11 test subjects, thresholds tuned on validation data.

## 3. What this means for this project

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
   multimodal model scores 0.116. Our near-chance six-session result
   (`docs/experiments/05_six_session_run.md`) is in line with this, not an outlier.
2. **Always report the trivial baseline.** Every honest result above is read against "predict
   the mean" or "predict the majority". We already report `majority`; for a continuous target
   (direction 3 of `05_six_session_run.md` §7) add a predict-the-mean MAE.
3. **Report against human agreement.** Labeler-vs-consensus agreement gives a ceiling; Petković et
   al. (VISAPP 2026, on their own data) report their model (r = 0.53) next to inter-rater ICC 0.85,
   a good template.
4. **Averaged labels make a low-variance target**, which shrinks MAE without the model learning
   much. Prefer correlation (Pearson or Spearman, and our `retention_r`) alongside MAE.
5. **Pose and head orientation carry most of the signal** where measured (§2.5). Both can be
   estimated from context cameras; head pose is a candidate feature family for us.
6. **Smartwatch data added nothing** in three independent works; not using it costs little.
7. **Frame rate matters little:** 1 fps equalled 10 fps for prediction (§2.3), consistent with
   labels annotated at one frame per second.

**How to present our results:** as the first context-camera, group-held-out benchmark on DIPSER,
with its own baselines (majority, predict-the-mean, hand-weighted) and human agreement as reference,
and an explicit note that per-student desk-camera numbers measure a different, easier task.

## Not verified

The full text of Liu et al. (paywalled) and of IPA 2.0 (access refused); about six works that
Google Scholar counts as citing DIPSER but Semantic Scholar does not list.
