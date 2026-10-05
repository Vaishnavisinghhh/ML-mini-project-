# Speaker notes (7 slides, about 8 minutes)

The same notes are in the slide notes of the PPTX.

1. **Title (Vaishnavi, 30 s).** Introduce the team and the problem: predict bankruptcy from financial ratios. Only 3.23% of companies are bankrupt, so accuracy is misleading.
2. **Dataset and EDA (Vaishnavi, 1.5 min).** Kaggle dataset, 6,819 companies, 95 ratios, 220 bankrupt. No missing values, one constant column dropped, 23 near-duplicate feature pairs. Strongest single feature has |r| of 0.315, so a multivariate model is needed.
3. **Approach (Yashavantha, 1.5 min).** Pipelines fitted on training data only. Six models. Imbalance handled by class weights, scale_pos_weight, and oversampling inside the MLP's fit. Models chosen by 5-fold CV on the training split, tested once on the held-out split.
4. **Results (Yashavantha, 1.5 min).** LightGBM first in CV and test (F1 0.525, PR-AUC 0.553). XGBoost: more recall, less precision. Naive Bayes fails. Oversampling lifted the MLP's recall from 0.07 to 0.45.
5. **Live demo (Vaishnavi, 2 min).** Run `python -m src.demo`, then `--index 84` (missed bankruptcy, P = 0.001) and `--index 97` (false alarm, P = 0.604). The demo uses held-out test rows on purpose. Be honest about the failures.
6. **Conclusions (Yashavantha, 1 min).** Pick by CV, judge on the bankrupt class, choose the threshold by business cost. Limitations: only 44 positive test cases, no extensive tuning, 0.5 threshold, no time-based validation.
7. **Contributions (both, 30 s).** Each person states their own part; the commit history matches.

Before the review: run `python -m src.train` once so you know it works and takes about a minute, and rehearse the demo commands.
