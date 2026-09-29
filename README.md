# Evaluating Fairness and Mitigating Bias in Machine Learning: A Novel Technique using Tensor Data and Bayesian Regression

## Abstract
Fairness is a critical component of Trustworthy AI. In this paper, we focus on Machine Learning (ML) and the performance of model predictions when dealing with skin color. Unlike other sensitive attributes, the nature of skin color differs significantly. In computer vision, skin color is represented as tensor data rather than categorical values or single numerical points. However, much of the research on fairness across sensitive groups has focused on categorical features such as gender and race. This paper introduces a new technique for evaluating fairness in ML for image classification tasks, specifically without the use of annotation. To address the limitations of prior work, we handle tensor data, like skin color, without classifying it rigidly. Instead, we convert it into probability distributions and apply statistical distance measures. This novel approach allows us to capture fine-grained nuances in fairness both within and across what would traditionally be considered distinct groups. Additionally, we propose an innovative training method to mitigate the latent biases present in conventional skin tone categorization. This method leverages color distance estimates calculated through Bayesian regression with polynomial functions, ensuring a more nuanced and equitable treatment of skin color in ML models.


## Requirement
The Python dependencies required to run the code are listed in requirements.txt. Install them with: pip install -r requirements.txt

## Bayesian binomial spline analysis

Run `python bayesian.py` to pool all configured seeds' `valid.csv` predictions
for each dataset, training mode, and model in `config.py`. Test predictions are
excluded. Each prediction row has equal weight, including repeated images across
seeds. Within each true class, WD quantile bins provide correct and total counts.

`bayesian_spline.py` fits a binomial likelihood with a logit link and cubic
B-spline basis (a piecewise cubic curve, not one global cubic polynomial).
The defaults in `bayesian.py` are 25 bins, 7 uniformly spaced knots, a second-
difference prior precision of 10, and a coefficient prior standard deviation of 5.
These hyperparameters are fixed defaults, not selected by validation.
Inference uses a Laplace approximation around the posterior mode, with posterior
mean probabilities integrated by Gaussian quadrature. No additional dependencies
beyond `requirements.txt` are needed.

Outputs in `skin/{dataset}/bayes/` have the prefix
`{train_mode}_{model}_pooled_seeds_binomial_spline_`:

- `{class}.pkl`: fitted model, covariance, diagnostics, settings, and source seeds.
- `{class}_bins.csv`: mean WD, correct count, total count, and observed accuracy.
- `{class}_curve.csv`: posterior mean and pointwise approximate 95% credible bounds.
- `bayesian_regression.png`: class curves with credible bands and count-sized points.

The pickle retains `brr_poly.predict(X)` for compatibility with `BRFairLoss`;
`brr_poly.predict_interval(X)` returns the new bounds. `mu_value` remains the
uniform-WD-grid mean probability, not sample-weighted observed accuracy.
Predictions outside the fitted WD range use constant boundary extrapolation.
Previous polynomial outputs are preserved under their original filenames.

The bands describe uncertainty in the probability curve, not future observed
accuracy or simultaneous coverage of the entire curve. They condition on the
chosen bins, knots, and prior and assume independent binomial trials.
Repeated-image and seed dependence is not modeled, so the bands can be too narrow;
they should not be interpreted as uncertainty across independent patients/seeds.

Tests: `python -m unittest discover -s tests -p test_bayesian_spline.py -v`.
