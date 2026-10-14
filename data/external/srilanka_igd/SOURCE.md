# Source: Sri Lankan Internet Gaming Disorder dataset

**File:** `IGD_Database.xlsx` (unmodified copy of the published file)

**Citation:** Manchanayake, M. (2021). *Findings from a study investigating the prevalence and
associations of Internet Gaming Disorder among an Advanced Level student population from
Colombo, Sri Lanka* (Version 1) [Data set]. Mendeley Data. https://doi.org/10.17632/8r2jgm6ygh.1

**Related article:** The Nine-Item Internet Gaming Disorder Scale (IGDS9-SF): Its psychometric
properties among Sri Lankan students and measurement invariance across Sri Lanka, Turkey,
Australia, and the USA. *Healthcare*, 10(3), 490 (2022). https://doi.org/10.3390/healthcare10030490

**Licence:** Creative Commons Attribution 4.0 (CC BY 4.0) — reuse permitted with attribution.

**Contents:** 395 Advanced Level students from four schools in the Colombo Educational Zone
(cross-sectional, self-administered questionnaire). 322 are gamers with complete IGDS9-SF item
responses. Also includes gaming habits (weekday/weekend hours, game type, genre, device, starting
age), sleep hours and quality, demographics, psychosocial items and the Motives for Online Gaming
Questionnaire (MOGQ).

**Data issues found and how they are handled** (see `igd_features.load_dataset`):
- `MOGQ 6` and `MOGQ 19` each contain one out-of-range value (22, 21 on a 1–5 scale) → set to missing.
- The raw columns `Parent-Child`, `Teacher-Student` and `Appearance` duplicate the sleep columns
  (apparent column shift); they are not used.
- Hours and sleep are recorded as ranges; each range is represented by its midpoint.
- Column names contain trailing spaces; they are stripped on load.
