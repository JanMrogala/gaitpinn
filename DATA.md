# Data

What the study needs, in order:

1. Walking **and** stairs from the same people.
2. True sagittal hip, knee and ankle angles from motion capture.
3. Gyros on **both** thigh and shank, for the gyro physics term.
4. EMG of the knee flexors and extensors, for the swing physics term.
5. Height and mass.
6. Bonus: several sessions per person.

## How this was checked (Nov 2026)

- **Sources:** each dataset was checked against its paper and its data repository, with file
  listings, sizes and licence fields read from the repository APIs. For most datasets, single
  data files were opened through HTTP range requests.
- **Second pass:** an independent re-check repeated this for every dataset not rated poor. It
  focused on the claims that decide fit: stairs, angle source, IMU segments, EMG and sessions.
- **Weaker evidence, marked † in the table:**
  - Camargo: the journal full text was not accessible, so the methods were checked against the
    first author's dissertation (Georgia Tech, 2021, Ch. 3).
  - Lencioni: no data file could be opened, because figshare blocked automated download (403).
  - ENABL3S and MyPredict: the papers were read through a summarizing fetch, not verbatim.
  - Several datasets: only 1–3 people were opened.
- **Sessions marked "1 (inferred)":** the sources don't say "one day" outright. One session is
  inferred from the protocol and the file structure.

## Comparison

| Dataset | Access, licence, size | People | Walking / stairs | EMG | IMU (gyro on thigh + shank?) | True angles | Force plates | Sessions | Fit |
|---|---|---|---|---|---|---|---|---|---|
| **Camargo 2021**, J. Biomech., doi:10.1016/j.jbiomech.2021.110320 (Mendeley, 3 parts)† | Open, CC BY 4.0, 24.5 GB (single files range-readable) | 22 | Treadmill (28 speeds) and level-ground circuits; 35–45 stair files per person (4 step heights, up and down, approach and transitions inside each file); ramps | 11 muscles, right leg, 1000 Hz, band-passed raw: VM, VL, RF, BF, ST, gracilis, MG, TA, soleus, glut. med., ext. oblique | Dedicated foot, shank, thigh, trunk (6-axis, 200 Hz), right leg. **Yes** | Vicon + OpenSim IK, 200 Hz, hip/knee/ankle, both legs (goniometers also present) | Yes (Bertec) | 1 | **Strong** |
| **ENABL3S**, Hu, Rouse & Hargrove 2018, Front. Robot. AI, doi:10.3389/frobt.2018.00014 (figshare)† | Open, CC BY 4.0, 6.5 GB | 10 | Free circuits: walk, ramps, stairs, continuous transitions | 7 per leg, both legs (VL, RF, BF, ST, TA, MG, SOL) | Thigh and shank (both legs), waist; no foot. **Yes** | **Goniometers only**, knee and ankle; **no hip** | No | 1 (inferred) | Possible |
| **Dimitrov 2023**, Sci. Data, doi:10.1038/s41597-023-02679-x (figshare) | Open, CC BY 4.0, 2.8 GB | 10 | Walking in session 1; stairs and ramps in session 2 | One 64-ch HD grid below the knee only (no thigh muscles) | **One** IMU, right shank. **No thigh gyro** | Vicon + OpenSim IK | Yes (2) | 2, with different tasks in each | Poor |
| **Lencioni 2019**, Sci. Data, doi:10.1038/s41597-019-0323-z (figshare)† | Open, CC0 / CC BY (sources disagree), 0.26 GB | 50 (aged 6–72) | One published stride per trial; step up/down; no continuous switch | 8, dominant leg | **None** | Marker-based (LAMB model) | Yes (2) | 1 | Poor |
| **Scherpereel 2023**, Sci. Data, doi:10.1038/s41597-023-02840-6 (GT repository) | Open, CC BY 4.0, 79 GB | 12 | Treadmill walking; short stair passes (~3 s each, ~32 s per person) | 8 per leg, both legs | Built into EMG sensors (shank, front and back thigh, pelvis). Yes, but no dedicated segment IMU | Vicon + OpenSim IK | Yes | 1 | Possible (height not found) |
| **SIAT-LLMD**, Sci. Data, doi:10.1038/s41597-023-02263-3 (figshare) | Open, CC0, 8.4 GB | 40 | Walking; ~1.5 s trimmed stair trials | 9, left leg (no BF) | **None** | Mocap + OpenSim IK (60 Hz) | Torques only | 1 | Poor |
| **Darmstadt stairs**, Grimmer et al. 2023, TUdatalib doi:10.48328/tudatalib-1182 | Open, CC BY 4.0, 8.8 GB (3.8 GB needed) | 12 (all male) | 60 trials per person: short level walking → stairs up → top → stairs down → level; 3 step heights | 6 per leg, both legs (RF, VL, BF, TA, GAS, SOL; only flexor is BF) | IMU in every EMG sensor; dedicated shank and foot; **thigh only via EMG sensors on muscle bellies** | Qualisys + OpenSim IK | Yes (7) | 1 (inferred) | Possible |
| **MyPredict 3**, Schulte et al. 2023, Sci. Data, doi:10.1038/s41597-023-02341-6 (4TU)† | Open, CC BY 4.0, 184 GB (range-readable HDF5) | 10 | 40 circuits per day: walk → stairs up → … → stairs down → walk | 8 per leg, both legs | Xsens on foot, shank, thigh (both legs), pelvis, T8. **Yes** | **Xsens, computed from the same IMUs**, not optical | No | **4 days (1, 2, 3, 7)** | Possible |
| **Brantley 2018**, Sci. Data, doi:10.1038/sdata.2018.133 (figshare) | Open, CC BY 4.0, 16.8 GB (5.5 GB needed) | 10 | Continuous course: walk, ramps, stairs | 6 per leg, both legs | Xsens full body. **Yes** | **Xsens**, not optical | No | 1 (inferred) | Possible (no height or mass) |
| **Gait120**, Boo et al. 2025, Sci. Data, doi:10.1038/s41597-025-05391-0 (figshare) | Open, CC BY 4.0, 16.5 GB | 120 (all male) | Separate cropped ~3 s trials, no continuous switch | 12, right leg | **None** | Vicon + OpenSim IK | Walking only | 1 (inferred) | Poor |
| **VUB daily life**, Baetens et al., Figshare+ doi:10.25452/figshare.plus.30294658 | Open, CC0, 141 GB | 20 | Many daily activities incl. stairs | 8 | Only in foot insoles. **No thigh or shank gyro** | Vicon Plug-in Gait | Yes (4) | 1 | Poor |
| **K2MUSE**, Li et al., Int. J. Robot. Res. (Kaggle) | Open, **CC BY-NC 4.0**, 151 GB | 42 | Treadmill only, **no stairs** | 13 | **None** | Vicon Plug-in Gait | Yes | 8 people have a second day (walking only) | Poor |

### Found by the search, but only checked in a first pass

The search found 15 new candidates, and only the 6 best were verified in depth. The other 9
got one search-stage check each. Gaps found there:

| Dataset | Several sessions? | Main gap |
|---|---|---|
| NONAN GaitPrint (Sci. Data 2023, doi:10.1038/s41597-023-02704-z) | **Yes**, 2 days a week apart | Level walking only, no stairs, no EMG; IMU-derived angles |
| Jayasinghe et al. (loose-clothing IMUs) | **Yes**, 1–4 days, 5 people | IMUs on loose trousers; no joint angles, no EMG |
| HuGaDB | No (17 of 18 people on one day) | No joint angles; 2 EMG channels; many corrupted gyro files |
| Reznick et al. 2021 (Gregg lab) | 1 (inferred) | No IMU, no EMG |
| Hossain et al. (WMCG) | 1 (inferred) | No EMG released; separate stair trials |
| Sharma et al. (non-laboratory gait) | Mostly no | No EMG; Xsens angles |
| Losing & Hasenjäger 2022 | Unclear | No EMG; Xsens angles |
| LocoD (Chalmers) | 1 | No joint angles at all |
| MEILoD | Unknown | No joint angles; EMG only on thigh muscles |

**Several sessions per person:** MyPredict 3 is the only candidate found that repeats walking
**and** stairs on different days with EMG and thigh and shank gyros. Its target angles come from
the same Xsens IMUs used as inputs, so they are not an independent ground truth. NONAN GaitPrint
and Jayasinghe are multi-day too, but they lack stairs or angles.

## Recommendation: Camargo 2021

Several datasets meet needs 1–5 on paper: Camargo, Darmstadt and Scherpereel, the last except
for height. Camargo is the strongest of them because it is the only one with all of these:

- **Dedicated IMUs** on thigh, shank and foot. Darmstadt has no dedicated thigh IMU; Scherpereel
  has no dedicated segment IMUs at all.
- **Plenty of steady walking** for training. Treadmill at 28 speeds plus level-ground circuits,
  against short, non-steady bouts in Darmstadt and 3 treadmill speeds in Scherpereel.
- **The most people:** 22, against 12 for each of the other two.
- **About 40 stair files per person.**
- **11 EMG channels**, with three knee extensors and four knee flexors.

### Caveats (to handle in the data step)

- **One session per person**, so session-to-session drift can't be tested with Camargo.
  MyPredict 3 is the candidate for that later.
- **Sensors on the right leg only.** The target is the right hip, knee and ankle (`*_r` columns
  in the IK table).
- **The gyro axis and sign differ per person.** The best axis for
  `d(knee)/dt ≈ gyro_shank − gyro_thigh` is Y in one person and a Y/Z mix in others (3 people
  checked). Alignment must be fitted per person on walking training data.
- **Stair files are short.** Each is ~15 s and contains:
  1. a 2–3 s walking approach;
  2. an ascent of 1.3–3 s;
  3. a 3–6 s stand and turn on the top platform;
  4. a descent of 1.2–2.8 s;
  5. about one stride of walk-off.

  The files are cut from longer recordings, and the 3–8 s between repetitions are not released.
- **The `idle` label at the start of stair files is already walking.** Use pelvis speed or gait
  events, not labels, to separate walking from standing.
- **The files are MATLAB tables**, which scipy can't read directly. They need a one-off
  conversion.
- **EMG is band-passed raw (20–400 Hz).** It must be rectified, smoothed with causal filters and
  resampled.

## Choice

**Camargo 2021**, chosen on 2026-11-19.
