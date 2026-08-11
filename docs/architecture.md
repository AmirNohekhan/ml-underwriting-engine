# Architecture

```mermaid
flowchart LR
    A["Synthetic portfolio generator"] --> B["Validation and feature table"]
    B --> C["Group-aware cross-validation"]
    C --> D["Tweedie GLM"]
    C --> E["XGBoost Tweedie"]
    D --> F["Model selection"]
    E --> F
    F --> G["Holdout calibration"]
    G --> H["Risk tiers and technical premium"]
    H --> I["Fairness and stability diagnostics"]
    H --> J["Drift and performance monitoring"]
    G --> K["FastAPI inference service"]
    C --> L["MLflow tracking"]
```

The project intentionally separates statistical modeling from decisioning. The
model predicts expected loss; pricing and underwriting tiers are a governed
decision layer that can be reviewed, constrained, and monitored independently.
