## The Calibration Problem

Modern neural networks are often **overconfident**: they assign high probabilities even when wrong. A model might say "95% confident" but actually be correct only 70% of the time at that confidence level.

Calibration methods fix this by learning a mapping from raw model outputs to well-calibrated probabilities. **Isotonic regression** is a non-parametric approach that learns this mapping from data.

---

## What Is Isotonic Regression?

Isotonic regression fits a **monotonically non-decreasing** function to data. If $x_1 < x_2$, then the fitted value $f(x_1) \leq f(x_2)$.

For calibration, this means: if the model is more confident about prediction A than prediction B, the calibrated probability of A should be at least as high as B.

The monotonicity constraint makes sense because higher raw confidence should map to higher calibrated confidence (or stay the same), never lower.

---

## The Isotonic Calibration Process

**Input:**
- Raw model predictions (probabilities or scores) on a calibration set
- True labels (0 or 1 for binary classification)

**Output:**
- A step function that maps raw predictions to calibrated probabilities

**Step 1:** Sort samples by raw prediction (ascending)

**Step 2:** Fit an isotonic regression: find the monotonically non-decreasing function that best predicts the true labels from the raw predictions

**Step 3:** Use this function to transform future predictions

---

## How Isotonic Regression Works

Given sorted raw predictions $s_1 \leq s_2 \leq ... \leq s_n$ and corresponding labels $y_1, y_2, ..., y_n$:

Find values $f_1, f_2, ..., f_n$ that:
1. Minimize $\sum_i (f_i - y_i)^2$ (squared error)
2. Subject to $f_1 \leq f_2 \leq ... \leq f_n$ (monotonicity)

This is solved by the **Pool Adjacent Violators Algorithm (PAVA)**.

---

## The PAVA Algorithm

**Intuition:** Start with the raw averages in each "block." When a block has a higher average than the next block (violating monotonicity), merge them and take the combined average.

**Step 1:** Initialize each point as its own block with value = label (0 or 1)

**Step 2:** Scan left to right. If block $i$ has value greater than block $i+1$:
- Merge the blocks
- Set the merged block's value to the weighted average
- Go back and check if the merged block violates monotonicity with its predecessor

**Step 3:** Repeat until no violations remain

The result is a step function: consecutive samples with the same fitted value form a "step."

---

## Worked Example

**Data (sorted by raw prediction):**

Raw predictions: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

Labels: [0, 0, 1, 0, 1, 0, 1, 1]

**Initial blocks:** Each point is its own block with value = label

[0, 0, 1, 0, 1, 0, 1, 1]

**PAVA iterations:**

Block 3 (value 1) > Block 4 (value 0): Merge into block with average 0.5

[0, 0, 0.5, 0.5, 1, 0, 1, 1]

Block 5 (value 1) > Block 6 (value 0): Merge into 0.5

[0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1]

Now check backwards: Block 4 (0.5) = Block 5 (0.5), OK

All monotonic. Final calibrated values:

[0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1]

---

## The Step Function

Isotonic regression produces a **step function**:

- Raw prediction in [0.1, 0.2]: calibrated to 0
- Raw prediction in [0.3, 0.6]: calibrated to 0.5
- Raw prediction in [0.7, 0.8]: calibrated to 1

For new predictions, find which interval they fall into and return the corresponding calibrated value.

---

## Interpolation for New Predictions

The fitted isotonic function is defined at the training points. For new predictions:

**Option 1: Nearest neighbor**
Find the closest training point and use its calibrated value.

**Option 2: Linear interpolation**
Interpolate between adjacent training points.

**Option 3: Clip to range**
If new prediction is outside training range, clip to the nearest endpoint value.

---

## Isotonic vs. Platt Scaling

**Platt scaling (parametric):**
- Fits a logistic regression: $P(y=1) = \frac{1}{1 + e^{-(As + B)}}$
- Learns only 2 parameters (A and B)
- Assumes sigmoid shape is appropriate
- Works well with limited calibration data

**Isotonic regression (non-parametric):**
- No assumed functional form
- Can model arbitrary monotonic relationships
- More flexible, can capture complex miscalibration patterns
- Needs more data to avoid overfitting

**When to use which:**

Isotonic: Larger calibration sets (1000+ samples), complex miscalibration patterns

Platt: Smaller calibration sets, when sigmoid assumption is reasonable

---

## Multi-Class Calibration

For multi-class problems, apply isotonic calibration to each class separately:

**One-vs-all approach:**
1. For each class $k$, collect the predicted probability $P(y=k)$ and the binary label (1 if true class is $k$, else 0)
2. Fit isotonic regression for each class
3. Calibrate each class probability independently
4. Optionally renormalize so probabilities sum to 1

---

## Advantages of Isotonic Calibration

**Flexibility:**
No assumptions about the shape of miscalibration. Can fix overconfidence, underconfidence, or non-monotonic patterns.

**Guaranteed monotonicity:**
Higher raw scores always map to equal or higher calibrated probabilities.

**Simplicity:**
PAVA is simple to implement and fast ($O(n)$ for $n$ samples).

**Non-parametric:**
Adapts to the data without assuming a functional form.

---

## Disadvantages and Pitfalls

**Overfitting:**
With small calibration sets, isotonic regression can overfit. The step function may be too jagged.

**Extrapolation:**
Isotonic regression does not extrapolate well beyond the training range. New predictions outside the range get clipped.

**Requires held-out data:**
You need a separate calibration set (not used for training the original model).

**Binning effect:**
The output is a step function, not smooth. This can be jarring if you expect continuous calibrated probabilities.

---

## Practical Implementation

**Step 1:** Split your data into train, calibration, and test sets

**Step 2:** Train your model on the train set

**Step 3:** Get raw predictions on the calibration set

**Step 4:** Fit isotonic regression: raw predictions (X) vs. true labels (y)

**Step 5:** Transform test set predictions using the fitted isotonic function

**Step 6:** Evaluate calibration on test set (e.g., compute ECE)

---

## Connection to Reliability Diagrams

Isotonic calibration is essentially fitting a step function to the reliability diagram. Where the reliability diagram shows (average confidence, accuracy) points, isotonic regression finds the monotonic function that best fits those points.