Smart WCET Insight Engine
Predict loop counts and Worst-Case Execution Time (WCET) from static code metrics using interpretable AI, with end-to-end explainability and a Streamlit-powered dashboard.

Project Overview
The Smart WCET Insight Engine is an AI-based analytics tool that:
Predicts loop bounds using static software code metrics.
Estimates WCET (Worst-Case Execution Time) using a two-stage regression pipeline.
Offers full SHAP explainability and interactive visualizations.
Deploys via a Streamlit dashboard supporting CSV uploads and batch predictions.

Dataset
Source: Code Metrics Dataset - Software Project Structure (Kaggle)
Rows: 509,426
Columns: 53
Features include: cbo, fanin, fanout, wmc, loc, lcom, loopQty, assignmentsQty, modifiers, and many more.
Missing values handled for: lcom*, tcc, lcc

Modeling Pipeline
Stage 1: Loop Quantity Prediction
Input: 49 static code metrics (excluding file, class, type, loopQty, WCET)
Best Model: DecisionTreeRegressor (R² = 0.9888)
Explainability: SHAP TreeExplainer with Waterfall and Beeswarm plots

Stage 2: WCET Prediction
Input: All features from Stage 1 plus predicted loopQty
Target: Simulated WCET = 3.5 * loopQty + 0.8 * assignmentsQty + noise
Best Model: DecisionTreeRegressor (R² = 0.9824)

Saved Models:
dt_loopQty_model.pkl
dt_wcet_model.pkl

Streamlit Dashboard
Features:
CSV Upload (under 25MB)
Data Preview
Prediction Output: Predicted_LoopQty, Predicted_WCET
Download predictions as CSV
SHAP Explainability:
Beeswarm plot (global impact)
Waterfall plot (case-wise impact)

How to Run
Clone the repository
git clone [https://github.com/SwaroopShivaraiTeli/GithubSmart-WCET-Insight-Engine]
cd smart-wcet-insight-engine

Install dependencies
pip install -r requirements.txt

Run the Streamlit dashboard
streamlit run smart_wcet_dashboard.py


Technologies Used
Python 3.12
Scikit-learn
SHAP
Matplotlib, Seaborn
Pandas, NumPy
Streamlit
XGBoost, Ridge, LinearRegression

Sample Input Format (CSV)
present in the folde names template_input

...
Notes
Only non-trivial methods with loops are included.

No manual loop annotation required.

WCET is generated synthetically using domain-informed logic.

Author
Swaroop S. Teli
MSc Data Science, Applied AI & Systems
www.linkedin.com/in/swaroop-teli-840896214
