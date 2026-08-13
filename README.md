# Food Delivery Time Prediction

An end-to-end MLOps project that predicts how long a food delivery will take, given order, restaurant, delivery-location, weather, and traffic details. Built as a resume/portfolio project, but the pipeline is real: raw data goes through cleaning, feature engineering, a documented train/test split, model experimentation tracked in MLflow, a versioned model registry with alias-based promotion, automated tests, CI, and a live deployed API.

**Live demo:** https://food-delivery-time-prediction-13z7.onrender.com/

This is deployed on Render's free tier, which spins the service down after a period of inactivity. If the demo has been idle, the first request can take 30-60 seconds while it cold-starts — that's Render waking the container up, not the model itself being slow. Give it a minute and it'll come back to life.

## What's actually in here

The dataset is a Swiggy (Indian food delivery) order log — restaurant and delivery coordinates, weather, traffic density, delivery person age/rating, and the actual time taken, per order. The target is `Time_taken(min)`, and this is a regression problem.

Everything downstream of the raw CSV is wired as a DVC pipeline (`dvc.yaml`), so the whole thing is reproducible with `dvc repro`. Hyperparameters, thresholds, and column configs live in `params.yaml`, not hardcoded in scripts — I wanted to be able to explain every number in this project, not just have it work.

## Pipeline stages

1. **data_cleaning** — the raw data has a handful of real problems that took some digging to find: missing values encoded as the literal string `"NaN "` (with a trailing space, which slips past pandas' default NA detection), a redundant `"conditions "` prefix baked into every weather value, some restaurant coordinates with flipped signs, a cluster of rows with all four lat/long values sitting at effectively (0,0) with no recoverable signal, and a "Metropolitian" typo in the source data itself. Also engineers `distance_km` (haversine between restaurant and delivery point) and time-of-day features from the order timestamp.
2. **data_preparation** — 80/20 train/test split, plus the one place in the pipeline where imputation is fit only on train (delivery person age imputed per-person from their own order history where available, ratings imputed with the train median) and applied to both splits, to avoid leaking test data into the imputation statistics.
3. **data_preprocessing** — ordinal encoding for the two genuinely ordered categorical columns (traffic density, number of simultaneous deliveries), one-hot encoding for the rest, and scaling — mainly for the linear baseline's benefit, since it's a no-op for the tree models.
4. **train** — trains the final selected model (tuned LightGBM) using the hyperparameters recorded in `params.yaml`.
5. **evaluation** — computes test MAE/R² and 5-fold CV MAE, writes `metrics.json`, logs everything to MLflow.
6. **register_model** — registers the trained model in MLflow's model registry and aliases it `staging`. Promotion to `production` is a separate, explicit step (see below), not something that happens automatically on every pipeline run.

## Model results

Every model here is compared against the same held-out test set and 5-fold CV, all logged to MLflow (hosted on DagsHub).

| Model | CV MAE (min) | Notes |
|---|---|---|
| Linear Regression (baseline) | 4.825 ± 0.026 | No tuning, just a reference point |
| Random Forest (tuned) | 3.142 ± 0.021 | RandomizedSearchCV, 30 trials |
| **LightGBM (tuned)** | **3.089 ± 0.017** | **Selected model** |
| Stacking (RF + LightGBM, LR meta-model) | 3.088 ± 0.017 | Evaluated, not promoted |

The stacking regressor's CV MAE is basically identical to LightGBM alone — the difference (0.001) is an order of magnitude smaller than the fold-to-fold standard deviation (0.017), so it's noise, not a real improvement. That's not surprising once you think about it: RF and LightGBM are both tree ensembles trained on the same features, so their predictions are highly correlated, which leaves a linear meta-model with almost nothing independent to combine. I kept the stacking run fully logged in MLflow as a documented comparison rather than deleting it — it's evidence the stacking approach was tried and genuinely didn't help, not something to hide.

I also ran an A/B test on target transformation (raw vs. `PowerTransformer`-wrapped) on the tuned LightGBM model, since the target's skew (0.482) was borderline in EDA. The raw target came out slightly better on both MAE and fold stability, so no transformation is applied.

## Model registry

The registered model (`food-delivery-time-lightgbm`) uses MLflow's alias-based registry API (`set_registered_model_alias`), not the deprecated stage-based one. Two aliases:

- `staging` — gets reassigned to whatever the pipeline most recently trained. Just means "latest candidate," not "good enough to serve."
- `production` — only moves here if the candidate's test MAE clears a threshold (3.14 min, derived as this model's own CV MAE mean + 3×CV std — i.e. a real statistical margin above measured noise, not a round number picked out of the air).

The FastAPI app always loads whatever's aliased `production`.

## Tests and CI

`tests/` has two things: registry health checks (does the staging model actually exist, load, and produce sane predictions) and the real gate — a test that loads the staging-aliased model and asserts its test MAE against the same threshold from `params.yaml` used for promotion, so the test and the promotion logic can never drift apart.

GitHub Actions (`.github/workflows/ci.yaml`) runs this test suite on pushes to `main` that touch anything pipeline-relevant, and only if it passes, promotes the staging model to production. It deliberately does `dvc pull` rather than `dvc repro` — this workflow's job is to validate whatever's already staged, not retrain on every push.

## Tech stack

- **Data/experiment versioning:** DVC + MLflow, both hosted on DagsHub (no AWS anywhere in this project — S3/ECR/CodeDeploy were never part of the plan)
- **Modeling:** scikit-learn, LightGBM
- **Serving:** FastAPI + Uvicorn
- **Deployment:** Render (Docker), free tier
- **CI:** GitHub Actions
- **Testing:** pytest

Full pinned versions are in `requirements.txt`.

## Project structure

```
src/
  data/
    data_cleaning.py       # raw -> cleaned, pre-split-safe transforms only
    data_preparation.py    # split + train-only imputation
  features/
    build_features.py      # time_period bucket
    preprocessing.py       # encoding + scaling
  models/
    train.py                       # trains the final LightGBM config
    evaluate.py                    # metrics + MLflow logging
    register_model.py              # registry + alias promotion
    train_baseline.py              # Phase 5 experiment, not in the pipeline
    train_random_forest.py         # Phase 6 experiment, not in the pipeline
    train_lightgbm.py              # Phase 6 experiment, not in the pipeline
    train_stacking.py              # Phase 6 experiment, not in the pipeline
    target_transform_experiment.py # Phase 6 A/B test, not in the pipeline
tests/
  conftest.py
  test_model_registry.py
  test_model_performance.py
notebooks/
  eda.ipynb
app.py                     # FastAPI app
static/demo.html           # plain HTML/JS demo form, served at /
dvc.yaml, dvc.lock          # pipeline definition
params.yaml                 # hyperparameters, thresholds, column config
Dockerfile
.github/workflows/ci.yaml
```

The `train_baseline.py` / `train_random_forest.py` / `train_lightgbm.py` / `train_stacking.py` / `target_transform_experiment.py` scripts are the actual experimentation history (Phases 5-6) — they're standalone, not wired into `dvc.yaml`, on purpose. The DVC pipeline reflects what ships (the one selected model), not the full trail of things I tried. The trail itself is preserved in MLflow, not deleted.

## How to reproduce this

1. Clone the repo, create a virtualenv, `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in your own DagsHub credentials (tracking URI is already correct for this project's DagsHub-hosted MLflow server; you need your own username/token to authenticate)
3. `dvc pull` to get the raw data and any tracked artifacts
4. `dvc repro` to run the full pipeline end to end — cleaning through registering the model as `staging`
5. `pytest tests/` to run the test suite against the staging model
6. If you want to promote it yourself: `python -m src.models.register_model --promote`
7. `uvicorn app:app --reload` to run the API locally, then open `http://127.0.0.1:8000/` for the demo form, or `http://127.0.0.1:8000/docs` for the raw API

To build and run the Docker image the same way it's deployed:
```
docker build -t food-delivery-time-prediction .
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=... -e MLFLOW_TRACKING_USERNAME=... -e MLFLOW_TRACKING_PASSWORD=... food-delivery-time-prediction
```
