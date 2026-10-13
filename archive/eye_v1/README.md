# Archived: first eye-fatigue models (retired 2026-10-13)

These files are kept for transparency. They are **no longer used by the app**.

Both models were trained on the MRL Eye Dataset (Fusek, 2018), whose file names are
`subject_image_gender_glasses_EYESTATE_reflections_lighting_sensor` (eye state is field 4:
0 = closed, 1 = open; field 5 is reflections: 0 none, 1 small, 2 big).

| Model | Problem found | Effect |
|---|---|---|
| Baseline CNN (`notebooks/04_eye_model_train.py`, "93.4% accuracy") | Read **field 5 (reflections)** as the eye state | It learned to detect "small reflection" images (7% of the data), not open vs closed eyes. Its 895 : 11,840 test split matches the reflection counts exactly. The 93.4% figure is invalid. |
| ResNet50 (`notebooks/05_resnet_eye_train.py`, 66.3% accuracy) | Correct label, but the **app read its output backwards** (it outputs P(closed); the app treated it as P(open)) | Open eyes scored as tired and closed eyes as alert. The model itself separated the classes only weakly (mean output 0.50 for open vs 0.62 for closed eyes). |

The app's single-photo score also mixed in fixed OpenCV thresholds on brightness, contrast and
whole-image redness, which were not validated.

## What replaced them

- `notebooks/08_train_eye_state_model.py`: eye-state CNN with the correct label and a
  **subject-wise** split (8 test people never seen in training): 94.9% accuracy, AUC 0.988.
- `eye_analysis.py`: YuNet face detector to locate both eyes, crops framed like the training
  images, minimum eye-size check, used by both the Eye Check and the background monitor.
- `notebooks/09_test_eye_check_on_photos.py`: test on ordinary colour photos.
