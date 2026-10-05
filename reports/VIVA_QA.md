# Likely viva questions (answers use this project's actual results)

**Why is accuracy a bad metric here?** Only 3.23% of companies are bankrupt. Predicting "healthy" for everyone gives 96.8% accuracy and catches nothing. We use recall, precision, F1 and PR-AUC on the bankrupt class.

**Why did LightGBM win?** It ranked first in 5-fold CV (F1 0.443) and on the test set (F1 0.525, PR-AUC 0.553). Boosted trees handle many correlated, skewed features and non-linear interactions without scaling. It found 21 of 44 bankrupt test companies with 15 false alarms.

**Why not just pick the model with the best test score?** That would make the test score optimistic. We select by cross-validation on the training split and use the test split once.

**How did you handle class imbalance?** Class weights (Logistic Regression, SVM, LightGBM), scale_pos_weight (XGBoost), and random minority oversampling inside the MLP's fit(). Naive Bayes has no such mechanism.

**Why oversample inside fit() and not before splitting?** Oversampling before the split copies minority rows into validation or test data and leaks information. Inside fit() it only touches the training part of each fold.

**Why did Naive Bayes get 6% accuracy?** It labels about 97% of companies bankrupt. It assumes features are independent, but 23 feature pairs have |r| above 0.95, so the same evidence is counted many times and probabilities for the bankrupt class are inflated.

**Why was the Neural Network poor at first?** MLPClassifier has no class_weight option, so it almost never predicted bankrupt (recall 0.07, CV F1 0.02). With oversampling, recall rose to 0.45 and CV F1 to 0.37.

**Why is XGBoost's recall higher than LightGBM's?** Its scale_pos_weight pushes it to flag more companies. It gets recall 0.73 but lower precision (0.39 vs 0.58). Which is better depends on the cost of a missed bankruptcy versus a false alarm.

**What does median imputation do here?** Nothing on this data, because there are no missing values. It is a safe default inside the pipeline.

**Why drop Net Income Flag?** It is constant, so it carries no information.

**How do you avoid data leakage?** The imputer and scaler are inside the Pipeline and fitted on training data only; the split is stratified with a fixed seed; the demo uses test rows only.

**What are the limitations?** One split with only 44 positive test cases, so metrics are noisy; limited tuning; 0.5 threshold; no time-based validation.

**How would you improve it?** Tune the threshold, calibrate probabilities, select features, search hyperparameters, and validate on a later time period.

**Why does the demo show a wrong prediction?** Row 84 is a bankrupt company the model scores 0.001; row 97 is a healthy one it scores 0.604. Showing them is honest and shows the recall-precision trade-off.
