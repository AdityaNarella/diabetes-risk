# Diabetes Risk Prediction (with Explainable AI)

An end-to-end ML project on the **PIMA Indians Diabetes** dataset. Includes:
- Data cleaning (fixing impossible zeros via imputation)
- Feature engineering (BMI category, Glucose/BMI ratio)
- Class imbalance handling with **SMOTE**
- Models: Logistic Regression, Random Forest, XGBoost
- Cross-validation and metric reporting (Accuracy, Precision, Recall, F1, ROC-AUC)
- Explainability using **SHAP**
- Exported pipeline (`joblib`) and optional **Streamlit** app

## 1) Project Structure
```
diabetes-risk/
├─ data/                      # put diabetes.csv here (PIMA dataset)
├─ notebooks/
│  └─ diabetes.ipynb          # main notebook with all steps
├─ src/
│  └─ streamlit_app.py        # simple UI to test the saved model
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## 2) Get the dataset
Download **PIMA Indians Diabetes** CSV as `data/diabetes.csv` with columns:
`Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome`

Sources: Kaggle (Pima Indians Diabetes Database) or UCI ML Repo.

## 3) Create & activate environment (choose one)

### Option A: venv (Windows PowerShell)
```powershell
cd diabetes-risk
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: venv (Linux/Mac)
```bash
cd diabetes-risk
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option C: Conda (any OS with Anaconda/Miniconda)
```bash
cd diabetes-risk
conda create -n diabml python=3.11 -y
conda activate diabml
pip install -r requirements.txt
```

## 4) Run the notebook
```bash
# from diabetes-risk (with env activated)
python -c "import webbrowser,sys; webbrowser.open('http://localhost:8888')"
jupyter notebook notebooks/diabetes.ipynb
```

Then run cells from top to bottom. The notebook will:
- Clean/prepare data
- Train LR/RF/XGB
- Evaluate and show plots
- Save best pipeline to `diabetes_risk_pipeline.joblib`

## 5) Run the Streamlit app (optional; after saving model)
```bash
streamlit run src/streamlit_app.py
```

Use the sidebar to enter patient values and see the predicted probability.

## 6) Push to GitHub (optional)
```bash
git init
git add .
git commit -m "Initial commit: diabetes-risk project"
# create a repo on GitHub named 'diabetes-risk', then:
git branch -M main
git remote add origin https://github.com/<your-username>/diabetes-risk.git
git push -u origin main
```

## 7) Resume bullets
- Built an **explainable** diabetes risk prediction pipeline (PIMA dataset) using Python, scikit-learn, XGBoost.
- Handled class imbalance with **SMOTE**, improved recall for diabetic cases.
- Interpreted model with **SHAP**; exported reusable pipeline and simple Streamlit UI.
