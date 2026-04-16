# 📉 Customer Churn Data — Exploratory Data Analysis (EDA)

A full exploratory data analysis on a telecom customer churn dataset to uncover **why customers leave**, **who is most at risk**, and **what business levers** can reduce churn.

---

## 📁 Project Structure

```
CUSTOMER-CHURN-DATA-EDA/
├── data/
│   └── Customer Churn.csv       # Raw dataset (7,043 rows × 21 columns)
├── analysis.ipynb               # Main EDA notebook
├── requirements.txt             # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📦 Dataset Overview

| Property | Value |
|---|---|
| **Source** | Telecom Customer Data |
| **Rows** | 7,043 customers |
| **Columns** | 21 features |
| **Target Variable** | `Churn` (Yes / No) |
| **Overall Churn Rate** | **26.54%** (1,869 churned out of 7,043) |

### Key Columns

| Column | Type | Description |
|---|---|---|
| `customerID` | String | Unique customer identifier |
| `gender` | Categorical | Male / Female |
| `SeniorCitizen` | Binary (0/1) | Whether customer is a senior citizen |
| `Partner` | Categorical | Has a partner or not |
| `Dependents` | Categorical | Has dependents or not |
| `tenure` | Numeric | Number of months the customer has been with the company |
| `PhoneService` | Categorical | Has phone service |
| `MultipleLines` | Categorical | Has multiple lines |
| `InternetService` | Categorical | DSL / Fiber Optic / No |
| `OnlineSecurity` | Categorical | Add-on: Online Security |
| `OnlineBackup` | Categorical | Add-on: Online Backup |
| `DeviceProtection` | Categorical | Add-on: Device Protection |
| `TechSupport` | Categorical | Add-on: Tech Support |
| `StreamingTV` | Categorical | Streaming TV service |
| `StreamingMovies` | Categorical | Streaming Movies service |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | Categorical | Uses paperless billing |
| `PaymentMethod` | Categorical | How the customer pays |
| `MonthlyCharges` | Float | Monthly bill amount |
| `TotalCharges` | Float | Total amount billed (lifetime) |
| `Churn` | Categorical | **Target** — Did the customer churn? |

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/IshanNaikele/CUSTOMER-CHURN-DATA-EDA.git
cd CUSTOMER-CHURN-DATA-EDA

# 2. Create & activate virtual environment
python -m venv my_env
source my_env/bin/activate        # Linux/Mac
my_env\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook analysis.ipynb
```

### `requirements.txt`
```
pandas
numpy
seaborn
matplotlib
jupyter
```

---

## 🔧 Data Cleaning

One key issue was found during the analysis:

> **`TotalCharges` column was stored as `string` (object) instead of `float`.**

This happened because some early-tenure customers had a blank space `" "` instead of `0` in that column. This was fixed as follows:

```python
data['TotalCharges'] = data['TotalCharges'].replace(" ", 0)
data['TotalCharges'] = data['TotalCharges'].astype(float)
```

After cleaning:
- ✅ No null values
- ✅ No duplicate `customerID` entries
- ✅ All columns are in correct data types

---

## 📊 Exploratory Analysis & Key Findings

---

### 1. 🎯 Churn Distribution

```
Churn: No  → 5,174 customers (73.46%)
Churn: Yes → 1,869 customers (26.54%)
```

> The dataset is **imbalanced** — about 3 customers stay for every 1 that leaves. This is important to keep in mind for any future machine learning work (use techniques like SMOTE or class weighting).

---

### 2. 💰 Total Charges Distribution

> *Histogram with KDE — `sns.histplot(data['TotalCharges'], kde=True)`*

The distribution of `TotalCharges` is **right-skewed**. This makes intuitive sense:

- The **majority of customers** are on low-cost or short-tenure plans — so most people cluster near the lower end of total spend.
- A **smaller group** of long-term, high-value customers have accumulated large total bills.
- The tail extends toward \$8,000+, representing loyal customers on premium plans.

> **Note:** Pie charts are inappropriate here because `TotalCharges` is continuous. A histogram with a KDE curve is the correct visualization choice.

---

### 3. 📦 Total Charges vs. Churn (Box Plot)

> *`sns.boxplot(x='Churn', y='TotalCharges', data=data)`*

| Metric | Churn: No | Churn: Yes |
|---|---|---|
| Median TotalCharges | ~\$1,684 | ~\$703 |

**Why this matters:**
- Customers who churned had **significantly lower total charges**, meaning they left *early* — before accumulating a large bill.
- This is not a coincidence. It tells us that **churn is an onboarding and early-experience problem**, not a long-term dissatisfaction problem.
- If customers were leaving due to service degradation over time, the "Yes" box would be *high* on the chart. It's not.

> 🔑 **Insight:** Winning back customers in the first few months is the highest-ROI churn-reduction strategy.

---

### 4. 🔵 Tenure vs. Total Charges (Scatter Plot)

> *`sns.scatterplot(x='tenure', y='TotalCharges', hue='Churn', data=data)`*

This scatter plot reveals a **fan/triangle shape**:

- The **top edge** = premium customers (high monthly charges × long tenure = very high total)
- The **bottom edge** = budget customers (low monthly charges × long tenure = moderate total)
- **Orange dots (Churn: Yes)** are almost entirely concentrated in the **bottom-left corner** (0–15 months)

```
Churned customers — Tenure stats:
  Mean tenure:   17.9 months
  Median tenure: 10 months

Retained customers — Tenure stats:
  Mean tenure:   37.6 months
  Median tenure: 38 months
```

> 🔑 **Insight:** Once a customer crosses ~20 months of tenure, they are very likely to stay. The first 20 months is the **danger zone**.

---

### 5. 📈 Monthly Charges vs. Churn (KDE Plot)

> *`sns.kdeplot(data=data, x='MonthlyCharges', hue='Churn', shade=True)`*

| Metric | Churn: No | Churn: Yes |
|---|---|---|
| Mean Monthly Charge | \$61.27 | **\$74.44** |

- There's a **massive blue peak near \$20/month** — low-cost plan customers almost never churn. They're stable and happy.
- The **orange (Churn) density rises sharply between \$70–\$110/month** — high monthly bills are a strong churn trigger.
- Customers start questioning value for money as bills exceed \$70–\$80/month.

> 🔑 **Insight:** Customers on higher-cost plans need extra engagement, loyalty rewards, or value-add features to justify their spend and reduce churn.

---

### 6. 🗺️ Correlation Heatmap

> *`sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='viridis')`*

| Pair | Correlation |
|---|---|
| `tenure` ↔ `TotalCharges` | **Strong positive** |
| `MonthlyCharges` ↔ `TotalCharges` | **Moderate positive** |
| `SeniorCitizen` ↔ anything | Weak |

The mathematical identity:
```
TotalCharges ≈ tenure × MonthlyCharges
```
is clearly visible in the heatmap — explaining the high correlation between those three variables. `SeniorCitizen` is nearly uncorrelated with all numeric features, meaning senior status alone doesn't predict spending behavior.

---

### 7. 👴 Senior Citizen vs. Churn

> *Stacked bar chart — normalized crosstab*

| Segment | Churn Rate |
|---|---|
| Non-Senior (0) | **23.6%** |
| Senior Citizen (1) | **41.7%** |

Senior citizens churn at nearly **double the rate** of non-seniors. Possible reasons:
- Seniors may be on more expensive plans relative to their perceived value.
- They may have less technical support or fewer digital service add-ons.
- They could be more price-sensitive.

> 🔑 **Insight:** Senior customers are a **high-risk segment** requiring dedicated retention programs or tailored plan offerings.

---

### 8. ⚧️ Gender vs. Churn

> *`sns.countplot(x="gender", hue='Churn', data=data)`*

```
Gender has NO statistically meaningful relationship with churn.
```

The churn rates for Male and Female customers are nearly identical. Gender should not be a factor in churn prediction models or targeted retention campaigns.

---

### 9. 💑 Partner Status vs. Churn

> *Stacked bar chart — normalized crosstab*

| Partner | Churn Rate |
|---|---|
| No Partner | **33.0%** |
| Has Partner | **19.7%** |

Customers with a partner churn significantly less. This is likely because:
- Partners may share plans (increasing switching cost).
- They may have more stable financial situations.
- Joint decision-making slows down impulsive cancellations.

> 🔑 **Insight:** Single customers are a higher-risk group. Offering family/partner bundles could improve retention.

---

### 10. 📋 Contract Type vs. Churn

> *`sns.countplot(x='Contract', hue='Churn', data=data)`*

| Contract Type | Churn Rate |
|---|---|
| Month-to-month | **42.7%** ⚠️ |
| One year | **11.3%** |
| Two year | **2.8%** ✅ |

This is one of the **strongest signals** in the entire dataset.

- Month-to-month customers have almost **no switching cost** — they can leave any month without penalty.
- Two-year contract customers almost never leave (2.8% churn rate).
- The relationship is clean and monotonic: longer commitment = lower churn.

> 🔑 **Insight:** Offering incentives to move customers from month-to-month to annual/biannual contracts is one of the **single most effective levers** for reducing churn.

---

### 11. 🌐 Internet Service vs. Churn

| Internet Service | Churn Rate |
|---|---|
| No Internet | **7.4%** |
| DSL | **19.0%** |
| Fiber Optic | **41.9%** ⚠️ |

Fiber Optic customers are leaving at an **extremely high rate**. This is concerning because Fiber is typically a premium product. Possible explanations:
- Fiber customers pay higher monthly charges (which we already know drives churn).
- Fiber customers may have higher expectations for reliability and speed.
- Competitors may be offering better fiber deals.

> 🔑 **Insight:** The Fiber Optic product either has a pricing or quality-perception problem that must be investigated urgently.

---

### 12. 💳 Payment Method vs. Churn

| Payment Method | Churn Rate |
|---|---|
| Bank transfer (automatic) | **16.7%** |
| Credit card (automatic) | **15.2%** |
| Mailed check | **19.1%** |
| Electronic check | **45.3%** ⚠️ |

Electronic check users churn at an alarming **45.3%** rate — nearly **3× higher** than automatic payment users.

Why? Customers who pay manually (especially via electronic check) are:
- Less "locked in" — no automatic renewal habit.
- More likely to notice their bill each month and reconsider.
- May have chosen this payment method because they were already skeptical about committing.

> 🔑 **Insight:** Nudging customers toward automatic payment methods (credit card or bank transfer) reduces friction and significantly lowers churn risk.

---

## 🧠 Summary of Key Insights

| # | Insight | Business Action |
|---|---|---|
| 1 | **26.54% churn rate** — 1 in 4 customers leaves | Set a baseline KPI to reduce this |
| 2 | **Short tenure = high churn** (median: 10 months for churned) | Invest heavily in onboarding (months 1–20) |
| 3 | **High monthly charges drive churn** (avg \$74 for churned vs \$61 for retained) | Offer loyalty discounts or value-add at \$70+ tiers |
| 4 | **Month-to-month contracts = 42.7% churn** | Incentivize annual/biannual plan upgrades |
| 5 | **Fiber Optic churn = 41.9%** | Investigate product quality + competitive pricing |
| 6 | **Electronic check churn = 45.3%** | Encourage auto-pay enrollment |
| 7 | **Senior citizens churn at 41.7%** | Create senior-specific support/retention programs |
| 8 | **Customers without partners churn at 33%** | Promote family/partner bundles |
| 9 | **Gender has no effect on churn** | Do not segment retention campaigns by gender |

---

## 🚀 Future Work

- [ ] Build a **Churn Prediction Model** (Logistic Regression, Random Forest, XGBoost)
- [ ] Handle **class imbalance** using SMOTE or class weighting
- [ ] Perform **feature engineering** (e.g., revenue per month bucket, service bundle score)
- [ ] Build a **Churn Risk Scoring Dashboard**
- [ ] Deploy model as an API for real-time churn scoring

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Pandas | Data loading, cleaning, transformation |
| NumPy | Numerical operations |
| Seaborn | Statistical visualizations |
| Matplotlib | Plot customization and rendering |
| Jupyter Notebook | Interactive analysis environment |

---

## 📄 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

---

> *"You can't fix churn you can't see. This EDA makes the invisible visible."*
