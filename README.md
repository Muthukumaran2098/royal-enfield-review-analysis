# Royal Enfield App Review Analysis

> End-to-end NLP pipeline to extract, classify, and prioritise user intent from Google Play Store reviews of the Royal Enfield RE-Prime app.

---

## Overview

Royal Enfield's RE-Prime app receives thousands of Play Store reviews — users flagging crashes, OTP failures, service booking issues, and more. At this volume, no team can manually read, categorise, and act on feedback at scale.

This project builds a structured NLP pipeline that:
- Scrapes real reviews from the India Play Store
- Classifies each review into **L1 topics** (what the user is talking about) and **L2 business impact** (why it matters)
- Scores every issue using a **Priority Matrix (P0–P3)** based on frequency and severity
- Delivers actionable, evidence-backed recommendations tied to real review data

---

## Repository Structure

```
Royal_Enfield_Review_Analysis/
│
├── Royal_Enfield_Review_Analysis.ipynb   # End-to-end notebook (scraping → insights)
├── reviews_raw.csv                        # Scraped dataset (2,543 reviews)
│
├── Project/                               # Modular pipeline scripts
│   ├── main.py                            # Pipeline entry point
│   ├── scraper.py                         # Play Store scraping logic
│   ├── classifier.py                      # L1/L2 keyword NLP classifier
│   ├── analyzer.py                        # EDA + priority matrix
│   ├── visualizations.py                  # Chart generation (matplotlib/seaborn)
│   └── requirements.txt                   # Python dependencies
│
├── data/
│   ├── raw/reviews_raw.csv                # Raw scraped reviews
│   └── processed/reviews_classified.csv  # Classified with L1/L2 labels
│
└── outputs/
    ├── Royal_Enfield_App_Review_Analysis.pptx   # Final presentation (7 slides)
    └── charts/                                   # 6 generated visualisations
        ├── 01_rating_distribution.png
        ├── 02_monthly_trend.png
        ├── 03_l1_intent_distribution.png
        ├── 04_l2_business_impact.png
        ├── 05_sentiment_by_topic.png
        └── 06_priority_matrix.png
```

---

## Dataset

| Attribute | Value |
|---|---|
| App | Royal Enfield RE-Prime (`com.royalenfield.reprime`) |
| Region | India (`en_IN`) |
| Reviews | 2,543 (after deduplication and length filtering) |
| Date Range | July 2022 — April 2026 |
| Average Rating | 2.29 / 5.0 |
| Negative Reviews (≤2★) | 64.1% |

---

## Classification System

### L1 Intent — What is the user talking about?

| Topic | Description | Negative % |
|---|---|---|
| Performance | Crashes, buffering, slow loading, freezing | 79% |
| Bugs | Login failures, firmware errors, broken flows | 76% |
| Customer Support | Delivery delays, no response, dealer issues | 63% |
| Billing | Payment failures, refund issues, RSA charges | 80% |
| Security | Unauthorised slot changes, OTP/data concerns | 73% |
| UI/UX | Confusing navigation, outdated design | 67% |
| Features | Missing trip tools, configurator, GPS tracking | 59% |

### L2 Impact — Why does it matter to the business?

| L1 Topic | L2 Business Impact |
|---|---|
| Performance, Bugs | User Retention |
| Customer Support, Security | Brand Trust |
| UI/UX, Features | Engagement |
| Billing | Revenue |

---

## Priority Matrix

Issues are scored using:

```
Priority Score = Frequency% × 0.6 + Impact Score × 0.4
```

Where `Impact Score = (5 − avg_rating) × 20`

| Priority | Threshold | Action |
|---|---|---|
| P0 | Score ≥ 60 | Fix immediately |
| P1 | Score ≥ 40 | Next sprint (4–6 weeks) |
| P2 | Score ≥ 20 | Backlog |
| P3 | Score < 20 | Monitor |

---

## Key Findings

- **1-star reviews (53.9%)** outnumber all positive reviews combined — sustained across 4 years
- **Performance + Bugs = 35.9%** of all reviews, both with 75%+ negative sentiment
- **Same complaints repeat across app versions** — systemic issues, not one-off regressions
- **26.6% of loyal users** (4-5★) cite the bike configurator and community as genuine strengths

---

## Quickstart

### Option 1 — Jupyter Notebook (recommended)

```bash
pip install -r Project/requirements.txt
jupyter notebook Royal_Enfield_Review_Analysis.ipynb
```

The notebook loads the pre-scraped `reviews_raw.csv` by default (`USE_CACHED_DATA = True`).  
Set it to `False` to re-scrape live from the Play Store.

### Option 2 — Pipeline Scripts

```bash
pip install -r Project/requirements.txt
python Project/main.py
```

This runs the full pipeline: scrape → classify → analyse → generate charts.

---

## Tech Stack

| Layer | Tool | Reason |
|---|---|---|
| Scraping | `google-play-scraper` | Reliable pagination, India region support |
| Data | `pandas`, `numpy` | Standard data manipulation |
| NLP | Keyword classifier (BERT-ready) | Interpretable, zero infrastructure for POC |
| Visualisation | `matplotlib`, `seaborn` | Publication-ready charts |
| Presentation | `pptxgenjs` | Programmatic slide generation |

---

## Production Scaling Roadmap

| Phase | Timeline | What |
|---|---|---|
| Data Infrastructure | Week 1–2 | Airflow DAG, cloud storage, daily scraping |
| ML Upgrade | Week 3–4 | Fine-tune DistilBERT (target: 90%+ accuracy) |
| Insights Platform | Week 5–8 | Live dashboard, Slack alerts on P0 spikes |

---

## Author

**Muthukumaran P** — Data Scientist

- Email: muthuwr1998@gmail.com
- GitHub: [github.com/Muthukumaran2098](https://github.com/Muthukumaran2098)
- LinkedIn: [linkedin.com/in/muthukumaran98](http://www.linkedin.com/in/muthukumaran98)
