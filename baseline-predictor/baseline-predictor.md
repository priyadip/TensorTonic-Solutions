## What Is a Baseline Predictor?

A baseline predictor in recommender systems estimates a user's rating for an item using only global statistics and user/item biases, without considering any similarity-based or latent factor information.

The baseline captures the expected rating based on:
- The overall average rating in the system
- How much this user tends to rate above or below average
- How much this item tends to be rated above or below average

---

## The Baseline Formula

$$
b_{ui} = \mu + b_u + b_i
$$

where:
- $b_{ui}$ is the baseline estimate for user $u$ rating item $i$
- $\mu$ is the global mean rating across all ratings
- $b_u$ is the user bias (user $u$'s deviation from global mean)
- $b_i$ is the item bias (item $i$'s deviation from global mean)

---

## Understanding Each Component

**Global mean $\mu$:**

The average of all ratings in the dataset.

$$
\mu = \frac{1}{|R|} \sum_{(u,i) \in R} r_{ui}
$$

**User bias $b_u$:**

How much user $u$ tends to rate above or below the global average. A user who rates generously has positive $b_u$; a harsh critic has negative $b_u$.

**Item bias $b_i$:**

How much item $i$ tends to be rated above or below average. Popular, high-quality items have positive $b_i$; disliked items have negative $b_i$.

---

## Computing User and Item Biases

**Simple approach (mean deviation):**

$$
b_u = \frac{1}{|I_u|} \sum_{i \in I_u} (r_{ui} - \mu)
$$

$$
b_i = \frac{1}{|U_i|} \sum_{u \in U_i} (r_{ui} - \mu)
$$

where $I_u$ is items rated by user $u$, and $U_i$ is users who rated item $i$.

**Problem:** This double-counts the bias. If a generous user rates a popular item, both get credit for the high rating.

---

## Iterative Bias Estimation

A better approach alternates between estimating user and item biases:

**Initialize:** $b_u = 0$ for all users, $b_i = 0$ for all items

**Iterate until convergence:**

1. Update item biases:

$$
b_i = \frac{\sum_{u \in U_i} (r_{ui} - \mu - b_u)}{|U_i|}
$$

2. Update user biases:

$$
b_u = \frac{\sum_{i \in I_u} (r_{ui} - \mu - b_i)}{|I_u|}
$$

This separates the effects of user generosity from item quality.

---

## Regularized Bias Estimation

To prevent overfitting, especially for users or items with few ratings, add regularization:

$$
b_i = \frac{\sum_{u \in U_i} (r_{ui} - \mu - b_u)}{\lambda_1 + |U_i|}
$$

$$
b_u = \frac{\sum_{i \in I_u} (r_{ui} - \mu - b_i)}{\lambda_2 + |I_u|}
$$

The regularization terms $\lambda_1$ and $\lambda_2$ shrink biases toward zero when there is little data.

---

## Worked Example

**Ratings:**

- User A rates Item 1: 5
- User A rates Item 2: 4
- User B rates Item 1: 3
- User B rates Item 2: 2
- User C rates Item 1: 4

**Step 1: Global mean**

$$
\mu = \frac{5 + 4 + 3 + 2 + 4}{5} = \frac{18}{5} = 3.6
$$

**Step 2: Simple bias calculation**

User A: Items rated = {1, 2}, ratings = {5, 4}

$$
b_A = \frac{(5-3.6) + (4-3.6)}{2} = \frac{1.4 + 0.4}{2} = 0.9
$$

User B: Items rated = {1, 2}, ratings = {3, 2}

$$
b_B = \frac{(3-3.6) + (2-3.6)}{2} = \frac{-0.6 + (-1.6)}{2} = -1.1
$$

User C: Items rated = {1}, rating = {4}

$$
b_C = \frac{4-3.6}{1} = 0.4
$$

Item 1: Users = {A, B, C}, ratings = {5, 3, 4}

$$
b_1 = \frac{(5-3.6) + (3-3.6) + (4-3.6)}{3} = \frac{1.4 - 0.6 + 0.4}{3} = 0.4
$$

Item 2: Users = {A, B}, ratings = {4, 2}

$$
b_2 = \frac{(4-3.6) + (2-3.6)}{2} = \frac{0.4 - 1.6}{2} = -0.6
$$

**Step 3: Baseline prediction**

Predict User C's rating for Item 2:

$$
b_{C,2} = \mu + b_C + b_2 = 3.6 + 0.4 + (-0.6) = 3.4
$$

---

## Why Baselines Matter

**They capture a lot of variance:**

In many datasets, baselines alone explain 50-70% of rating variance. Users and items have strong inherent biases.

**Foundation for other models:**

More sophisticated models often predict the residual $r_{ui} - b_{ui}$ rather than the raw rating. This makes the remaining signal easier to learn.

**Simple and fast:**

Baselines require only counting and averaging. They can be computed efficiently even for large datasets.

---

## Baseline in Matrix Factorization

Matrix factorization models often incorporate baselines:

$$
\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i
$$

The latent factors $\mathbf{p}_u$ and $\mathbf{q}_i$ model the residual after accounting for biases.

During SGD training, biases are updated alongside latent factors:

$$
b_u \leftarrow b_u + \eta (e_{ui} - \lambda b_u)
$$

$$
b_i \leftarrow b_i + \eta (e_{ui} - \lambda b_i)
$$

where $e_{ui} = r_{ui} - \hat{r}_{ui}$ is the prediction error.

---

## Baseline in Neighborhood Methods

Neighborhood-based collaborative filtering also uses baselines:

$$
\hat{r}_{ui} = b_{ui} + \frac{\sum_{j \in N(i;u)} s_{ij} (r_{uj} - b_{uj})}{\sum_{j \in N(i;u)} |s_{ij}|}
$$

This predicts the baseline plus a weighted average of how neighbors deviate from their baselines.

---

## Time-Aware Baselines

User and item biases can change over time:

**User bias drift:**

$$
b_u(t) = b_u + \alpha_u \cdot \text{sign}(t - t_u) \cdot |t - t_u|^\beta
$$

where $t_u$ is the user's mean rating time.

**Item bias drift:**

New items may have inflated ratings initially, then stabilize.

**Temporal binning:**

Compute separate biases for different time periods.

---

## Handling New Users and Items

**New user (no ratings):**

Set $b_u = 0$. Prediction becomes $\mu + b_i$.

**New item (no ratings):**

Set $b_i = 0$. Prediction becomes $\mu + b_u$.

**Cold start:**

Both biases are zero. Prediction is just $\mu$.

With regularization, biases naturally shrink to zero when data is sparse.

---

## Clipping Predictions

Baseline predictions can fall outside the valid rating range:

If ratings are 1-5 and $b_{ui} = 5.3$, clip to 5.

$$
\hat{r}_{ui} = \max(r_{min}, \min(r_{max}, b_{ui}))
$$

This ensures predictions are always in the valid range.

---

## Baseline as Benchmark

Any recommender system should beat the baseline predictor. If your sophisticated model performs worse than $\mu + b_u + b_i$, something is wrong.

**Typical RMSE improvements:**

- Global mean only: RMSE baseline
- With user/item biases: 10-20% improvement
- With collaborative filtering: additional 5-15% improvement

---

## Decomposing Ratings

The baseline model decomposes each rating:

$$
r_{ui} = \mu + b_u + b_i + \text{interaction} + \text{noise}
$$

- $\mu$: What is the typical rating?
- $b_u$: Is this user generous or harsh?
- $b_i$: Is this item generally liked or disliked?
- interaction: Does this specific user like this specific item?
- noise: Random variation

The baseline captures the first three components.