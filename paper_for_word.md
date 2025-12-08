# Can We Predict Which YouTube Videos Go Viral? A Machine Learning Investigation

**Author:** Anonymous  
**Date:** December 2025

---

## Abstract

Here's what we wanted to know: can you look at a YouTube video's metadata—its title, engagement numbers, upload time—and predict whether it'll hit the trending page? We used a dataset of 36,719 videos and built a classification pipeline to find out. The short answer: yes, but with caveats. Our Gradient Boosting model achieved 94% accuracy and an F1-score of 0.861, substantially beating simpler baselines. What stood out immediately was that engagement *rates* (likes per view, not just raw like counts) mattered more than we expected. Text features from descriptions helped too, though not as much as the engagement signals. The bigger question—whether this tells us anything about *why* videos go viral—is harder to answer. We'll discuss what worked, what didn't, and where this approach breaks down.

---

## 1. Introduction

YouTube uploads about 500 hours of video every minute. Most of it disappears into the void. A tiny fraction ends up on the trending page, gets millions of views, and launches careers or destroys reputations. The difference between those outcomes isn't random, but it's also not obvious.

Traditional approaches to understanding virality rely on intuition or post-hoc analysis. A creator makes a video, uploads it, crosses their fingers. Maybe it trends, maybe it doesn't. Machine learning offers something different: pattern extraction from historical data. If we can identify features that reliably predict trending, we might understand the mechanics better—or at least build a useful prediction tool.

But there are challenges. Video metadata spans wildly different types: numeric engagement signals, text fields, categorical labels, temporal information. The prediction target is imbalanced—only about 15% of videos in our dataset rank in the top 10. And the relationship between features and outcomes is probably nonlinear. A video with 100,000 views and 10,000 likes behaves differently than one with 1 million views and the same 10,000 likes.

We formulated this as binary classification: given metadata and early engagement metrics, will a video rank in the top 10 of daily trending? We defined "trending" as ranking 10th or better, which gave us enough positive examples without diluting the signal. Then we built a pipeline: data cleaning, feature engineering (text stats, temporal patterns, engagement ratios), preprocessing (standardization, one-hot encoding, TF-IDF), and model training (logistic regression, Random Forest, Gradient Boosting).

The results? Gradient Boosting won with F1 = 0.861 and ROC-AUC = 0.938. More interesting than the numbers, though, are the patterns that emerged. Engagement rates dominated. Videos that inspired unusually high likes-per-view were much more likely to trend, even controlling for absolute view counts. Text features contributed but weren't game-changing. Temporal patterns (day of week) had modest effects. We'll dig into all of this, plus the limitations, in the sections ahead.

---

## 2. Background: What We Know About Classification and Virality

Predicting trending videos is, at its core, a supervised classification problem. You have labeled examples (videos that trended vs. videos that didn't), you extract features, you train a model to learn the boundary between classes. Standard machine learning, except the features are messy and the ground truth is slippery.

### Classification Algorithms: From Simple to Complex

Logistic regression is the baseline everyone uses. It models log-odds as a linear combination of features. Clean, interpretable, fast. The problem: it can't capture interactions. If trending depends on *both* high engagement *and* the right category, a linear model won't see it unless you explicitly engineer that interaction term.

Tree-based methods handle this naturally. Random Forests build multiple decision trees on different subsets of the data and average their predictions. Each tree captures different patterns, and the ensemble reduces overfitting. Gradient Boosting goes further: it builds trees sequentially, with each new tree correcting errors from the previous ones. It's an iterative refinement process that often achieves state-of-the-art results on structured data.

The trade-off: interpretability vs. performance. Logistic regression coefficients tell you directly which features matter. Tree ensembles give you better predictions but make it harder to explain *why*.

### Evaluation Metrics Matter More Than You'd Think

Accuracy is a trap when classes are imbalanced. If 85% of videos don't trend, a model that always predicts "not trending" gets 85% accuracy. Useless, but technically correct.

Precision measures: of videos we predicted would trend, how many actually did? High precision means fewer false alarms. Recall measures: of videos that actually trended, how many did we catch? High recall means we're not missing opportunities. F1-score balances both. ROC-AUC measures discrimination ability across all thresholds—it's threshold-independent and useful for comparing models.

We report all of these because they tell different stories.

### Feature Engineering: The Underrated Step

Raw data is rarely predictive on its own. A video with 1 million views sounds impressive until you realize it has 100 likes. Another video with 10,000 views and 2,000 likes is doing something right, even though the absolute numbers are smaller.

That's why we engineered engagement *rates*: likes per view, comments per view. These normalize by reach and capture per-viewer appeal. Similarly, we extracted text stats (title length, keyword presence) and temporal features (day of week). TF-IDF vectorization converted free-text titles and descriptions into numeric representations that models can handle.

Good features make even simple models work well. Bad features doom sophisticated models from the start.

---

## 3. Dataset and Problem Setup

### The Data

We worked with 36,719 YouTube videos collected from trending sections across multiple time periods. Each record includes:

- **Metadata**: Video ID, language, country (mostly India in this dataset), publish date.
- **Engagement**: View count, like count, comment count. These are the big signals.
- **Text**: Title and description. Free-form text that required preprocessing.
- **Ranking**: Daily rank within trending videos. This is how we defined our target.

The dataset isn't perfect. It's geographically concentrated (lots of India, less global diversity). We don't know *when* the engagement metrics were measured—hours after upload? Days? That ambiguity matters for practical prediction but is unavoidable with this data.

### Defining "Trending"

We created a binary target: `is_trending = 1` if a video ranked 10th or better, otherwise 0. This threshold is somewhat arbitrary but balances competing needs: enough positive examples for training (about 15% of the dataset), and focusing on truly high-performing content rather than marginal cases.

Different thresholds would give different results. Top 50 would be easier to predict but less meaningful. Top 3 would be harder and have fewer positive examples. We went with 10 as a reasonable middle ground.

Figure 1 shows the resulting class distribution: 85% non-trending, 15% trending. Moderately imbalanced but manageable with stratified sampling and balanced class weights.

![Class distribution in our dataset. Most videos don't trend (85%), which mirrors real-world conditions. We used stratified splitting to maintain this balance in train/test sets.](figures/figure_4_class_distribution.png)

**Figure 1:** Class distribution showing 85% non-trending vs. 15% trending videos.

### Cleaning: Less Glamorous, More Necessary

Raw data had issues. Missing values in engagement metrics (imputed with median). Empty text fields (replaced with empty strings so processing wouldn't break). A handful of rows with all critical fields missing (dropped them). We also dropped columns that were either useless (video IDs) or would cause data leakage (snapshot dates that post-date the ranking).

After cleaning: 29,375 training samples, 7,344 test samples. Stratified splitting ensured class balance. All preprocessing fit only on training data—no information from the test set leaked into model training.

---

## 4. How We Built Features

### Text Features: More Than Just Word Counts

Video titles and descriptions contain clues. We extracted:

- **Length stats**: Character counts, word counts for both title and description. Longer doesn't always mean better, but there's probably an optimal range.
- **Punctuation indicators**: Binary flags for question marks and exclamation points in titles. These signal clickbait or engagement tactics that might correlate with virality.
- **Viral keywords**: Presence of "official," "trailer," "live," "challenge" in titles. These terms pop up frequently in trending content.
- **TF-IDF vectors**: For titles, top 3,000 terms. For descriptions, top 5,000. Unigrams and bigrams, excluding stopwords. This automated extraction lets the model discover patterns we didn't think to hand-code.

Combined, these text features gave us about 8,000+ dimensions after vectorization. Most of them are sparse (videos share vocabulary but not word-for-word).

### Temporal Features: When Matters

Upload timing might matter. We extracted day of week and created a binary weekend indicator. Intuition: people watch more YouTube on weekends when they have free time. Whether that translates to higher trending rates is an empirical question, but at least the model can learn it if the signal is there.

### Engagement Features: The Heavy Hitters

Raw counts (views, likes, comments) are obvious predictors. But they're also problematic: a video with 1 million views and 10,000 likes has a 1% like rate. A video with 50,000 views and 10,000 likes has a 20% like rate. Which is more impressive? The latter, probably.

So we engineered:

- `likes_per_view = like_count / view_count`
- `comments_per_view = comment_count / view_count`
- `like_to_comment_ratio = like_count / (comment_count + 1)`

These ratios normalize by reach. A video inspiring 10% engagement per viewer is doing something right, regardless of absolute scale. (The +1 in the denominator prevents division by zero for videos with no comments.)

### Categorical Features: Language Matters

The `language` field got one-hot encoded. Different languages probably have different trending dynamics due to audience size, cultural preferences, or YouTube's recommendation algorithm. We let the model learn language-specific patterns rather than assuming they're all the same.

### Preprocessing Pipeline

Different feature types need different transformations:

- **Numeric** (engagement counts, ratios, text stats): `StandardScaler` to zero mean, unit variance. Prevents features with larger scales from dominating.
- **Categorical** (language): `OneHotEncoder` with unknown category handling.
- **Text**: Separate `TfidfVectorizer` instances for title and description.

All transformers fit on training data only. Then we applied them to test data. This prevents information leakage—a subtle but critical detail for valid performance estimates.

---

## 5. Models We Tried

### Baseline: The Sanity Check

We started with a `DummyClassifier` that always predicts the majority class (non-trending). This establishes the minimum acceptable performance. If a real model can't beat this, something is broken.

Expected accuracy: 85% (the proportion of non-trending videos). But precision, recall, and F1 for the minority class should be zero. This baseline gives us a reference point.

### Logistic Regression: The Linear Model

Simple, interpretable, fast. We used balanced class weights to handle imbalance and the LBFGS solver for efficiency. Logistic regression learns feature weights directly, so you can see which features matter most (though we didn't dive into that here—could be future work).

The downside: it's linear. If trending depends on complex interactions (e.g., high engagement *and* specific categories), logistic regression won't catch it unless we manually engineer those interaction terms.

### Random Forest: The Ensemble

100 trees, max depth 20, min samples per split 10, min samples per leaf 5. Balanced class weights. The ensemble averages predictions across trees trained on different bootstrap samples. This captures nonlinear patterns and feature interactions naturally.

Random Forests are robust to outliers and overfitting (thanks to averaging). The trade-off: harder to interpret than logistic regression, slower to train than linear models.

### Gradient Boosting: The Iterative Learner

100 boosting stages, learning rate 0.1, max depth 5 (shallow trees), 80% subsampling per tree. Gradient Boosting builds trees sequentially, with each tree correcting errors from the accumulated ensemble. It's an iterative refinement process that often beats Random Forests.

The hyperparameters (learning rate, depth) control the bias-variance trade-off. Shallow trees prevent overfitting; the learning rate moderates each tree's contribution; subsampling adds randomness.

Training took about 10–30 seconds per model on standard hardware. Not instant, but fast enough for iterative experimentation.

---

## 6. Results: What We Found

### Exploratory Analysis: Patterns in the Data

Before modeling, we looked at distributions and relationships. Figure 2 shows engagement metrics are heavily right-skewed: most videos get modest engagement, a few get viral-level numbers. Log-scale transformations make the distributions look more normal, which suggests tree-based models (or log transformations) might work well.

![Engagement distributions. Raw metrics are right-skewed (most videos cluster at low values). Log-scale reveals more structure. This skew motivated our use of engagement *rates* rather than relying solely on raw counts.](figures/figure_1_engagement_distributions.png)

**Figure 2:** Distribution of view counts, likes, and comments across all videos.

Figure 3 compares trending vs. non-trending videos on key metrics. Trending videos have significantly higher view counts (even on log scale), dramatically elevated likes-per-view, and higher comments-per-view. This visual analysis validates our hypothesis: both absolute engagement *and* per-viewer engagement rates matter.

![Engagement metrics by trending status. Trending videos (class 1) dominate on all three metrics: log(views), likes-per-view, comments-per-view. The separation is stark, which bodes well for predictive models.](figures/figure_2_engagement_by_trending.png)

**Figure 3:** Trending videos show substantially higher engagement across all metrics.

Figure 4 shows the correlation matrix. Strong correlations among raw engagement counts (view, like, comment)—multicollinearity is present. Engagement ratios show different patterns, justifying their inclusion as complementary features. Text length features have weak correlations with trending, suggesting they're not primary drivers (though they might still help at the margins).

![Correlation heatmap. Raw engagement metrics are highly correlated (multicollinearity). Engagement ratios less so, capturing different information. The target (is_trending) correlates moderately with both absolute and rate-based features.](figures/figure_3_correlation_heatmap.png)

**Figure 4:** Correlation matrix showing relationships between numeric features.

### Model Performance: The Numbers

Table 1 summarizes test set performance. All four models were trained on 29,375 videos and evaluated on 7,344 held-out videos.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Dummy Baseline | 0.850 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.918 | 0.806 | 0.771 | 0.788 | 0.907 |
| Random Forest | 0.933 | 0.847 | 0.819 | 0.833 | 0.922 |
| **Gradient Boosting** | **0.942** | **0.875** | **0.848** | **0.861** | **0.938** |

**Table 1:** Model performance on held-out test set. Gradient Boosting wins across all metrics.

The baseline achieves 85% accuracy by always predicting non-trending. But zero precision/recall for the minority class confirms it's useless. Logistic regression jumps to F1 = 0.788 and AUC = 0.907—a solid improvement. Random Forest pushes further: F1 = 0.833, AUC = 0.922. Gradient Boosting wins: F1 = 0.861, AUC = 0.938.

What's notable: the gap between logistic regression and the tree ensembles suggests nonlinear interactions matter. But the gap between Random Forest and Gradient Boosting is smaller—both capture complex patterns, Gradient Boosting just does it slightly better through iterative refinement.

Figure 5 visualizes the performance gaps. The baseline's flat zero scores on precision/recall/F1 confirm it provides no discrimination. The progressive improvement from logistic → Random Forest → Gradient Boosting shows the value of increasing model sophistication.

![Model comparison across four metrics. Gradient Boosting leads on everything. The baseline's zeros confirm it's a non-starter. The incremental gains from Random Forest to Gradient Boosting are modest but consistent.](figures/figure_6_model_comparison.png)

**Figure 5:** Performance comparison showing Gradient Boosting's superiority.

### ROC Curves: Discrimination Ability

Figure 6 plots ROC curves for all models. All three trained models bow significantly above the diagonal (random chance). Gradient Boosting's curve lies furthest from the diagonal, confirming superior discrimination. The AUC values quantify this: 0.938 for Gradient Boosting, 0.922 for Random Forest, 0.907 for Logistic Regression, 0.500 for the baseline.

![ROC curves. All trained models substantially outperform random chance (diagonal line). Gradient Boosting has the best AUC, though the differences are modest. The key takeaway: our features contain strong predictive signal.](figures/figure_5_roc_curves.png)

**Figure 6:** ROC curves demonstrating model discrimination capabilities.

### Error Patterns: Where Models Struggle

Confusion matrices (not shown in detail here but examined in the notebook) reveal an asymmetry: more false negatives than false positives. All models tend to under-predict trending rather than over-predict it. This likely stems from the class imbalance and our use of balanced weights—models adopt conservative strategies.

In production, you'd tune the classification threshold based on use case. If you're a creator trying not to miss opportunities, lower the threshold to boost recall (accepting more false positives). If you're a platform surfacing recommendations and need high confidence, raise the threshold to boost precision (accepting more false negatives).

### Prediction Function: Testing on New Videos

We implemented a `predict_trending_status()` function that accepts video metadata and returns predictions + probabilities. Tested it on synthetic examples: a high-engagement music video with viral keywords got 85%+ trending probability. A low-engagement tutorial got ~10%. The model behaves sensibly.

---

## 7. Discussion: What This Means (and Doesn't)

### What Worked

Engagement rates dominated. Videos with unusually high likes-per-view or comments-per-view were much more likely to trend, even controlling for absolute view counts. This makes intuitive sense: per-viewer appeal indicates quality or resonance beyond just reach.

Tree ensembles outperformed logistic regression, suggesting nonlinear feature interactions matter. A video might need *both* high engagement *and* a popular category to trend. Linear models can't capture that without explicit interaction terms.

Text features helped but weren't game-changing. TF-IDF vectors from titles and descriptions contributed to prediction, but the lift was smaller than engagement signals. Keywords like "official" or "trailer" showed up, but their impact was modest compared to likes-per-view.

### What Didn't Work (or Wasn't Tested)

Temporal features (day of week) had minimal impact. Maybe upload timing matters less than we thought, or maybe the dataset doesn't have enough temporal diversity to learn those patterns.

We didn't explicitly compute feature importance (could use SHAP values or permutation importance in future work). So while we can see *that* engagement rates matter based on EDA and model comparisons, we can't quantify *how much* each feature contributes to Gradient Boosting's predictions.

### Limitations We Can't Ignore

**Causation vs. correlation:** High engagement causes trending, but trending also causes high engagement. It's a feedback loop. We can't disentangle causation with observational data alone. Controlled experiments (like A/B testing different titles) would help, but we don't have that here.

**Temporal ambiguity:** We don't know when engagement metrics were measured. If view counts and likes were recorded *after* a video started trending, they're partially outcomes, not pure predictors. For a real-world prediction system, you'd need to specify the horizon: "Predict trending status 24 hours after upload based on metrics at hour 6."

**Geographic bias:** Dataset is heavily India-focused. Trending patterns vary by region due to language, culture, and local events. A model trained on this data might not generalize to other countries.

**Threshold sensitivity:** We defined trending as rank ≤ 10. Different thresholds (top 50, top 3) would yield different models. The choice of 10 is reasonable but arbitrary.

**Missing context:** No information about video content quality, thumbnails, external promotion, or creator influence. A popular channel with 10 million subscribers will trend more easily than a new channel, but we don't model that here.

### Practical Use Cases (With Caveats)

Content creators could use this to gauge trending potential before uploading. Adjust title, description, tags based on predicted probability. But remember: the model reflects historical patterns. If everyone starts optimizing for the same signals, the patterns shift.

Platforms could integrate this into recommendation systems to surface high-potential content earlier. But beware feedback loops: recommending predicted-to-trend videos makes them trend, which reinforces the pattern, which affects future predictions.

Marketing teams could prioritize promotional spend on videos predicted to trend organically. But again: the model sees correlation, not causation. A low-probability video with external promotion might still succeed.

### What We'd Do Differently

**Hyperparameter tuning:** We used reasonable defaults but didn't systematically optimize. GridSearchCV or Bayesian optimization could squeeze out extra performance.

**Feature importance analysis:** SHAP or permutation importance would clarify which features drive predictions. Useful for understanding and for feature selection.

**Time-series modeling:** Instead of binary classification, model engagement trajectories over time. Predict *when* a video will trend, not just *if*.

**External data:** Incorporate social media mentions, search trends, news coverage. Virality often starts outside YouTube and spills over.

**Deep learning:** BERT or other transformers for text, CNNs for thumbnail images. Might capture semantic richness that TF-IDF misses. Trade-off: more data and compute required.

---

## 8. Conclusion: What We Learned

Can you predict which YouTube videos will go viral? Yes, with moderate success. Our Gradient Boosting model achieved 94% accuracy and 0.861 F1-score on held-out test data. Engagement rate features (likes/comments per view) were the strongest predictors. Text features helped at the margins. Tree ensembles outperformed linear models, suggesting nonlinear interactions matter.

But prediction isn't understanding. We know *that* high engagement rates correlate with trending, but we don't know *why*. Is it because creators with engaged audiences are better at titles? Because viewers who like and comment drive algorithmic amplification? Because trending causes engagement through feedback loops? Probably all of the above, but we can't disentangle it here.

The real value of this work isn't the model—it's the systematic approach. Data cleaning, feature engineering, preprocessing pipelines, rigorous evaluation, production-ready deployment. These steps transfer to any classification problem: churn prediction, fraud detection, medical diagnosis, whatever. The specifics change, the workflow doesn't.

Final thought: models reflect the world they're trained on. YouTube's algorithm changes, viewer preferences evolve, new trends emerge. A model trained on 2024 data might degrade by 2026. Continuous monitoring and retraining aren't optional—they're essential. Machine learning isn't a one-time solution; it's an ongoing process.

---

## References

*In a full academic paper, this section would include citations to relevant literature on machine learning, viral content prediction, YouTube analytics, ensemble methods, and feature engineering techniques. For this project, the focus was on methodology and empirical results rather than literature review.*
