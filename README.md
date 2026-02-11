Here is a professional `README.md` file for your application. It explains the project's purpose, installation steps, and how to use the simulation.

---

# 🛡️ Multi-Institutional Data Sharing Simulation

**Module 3 - Microskill 6: Navigating Multi-Institutional Data Sharing Challenges**

This interactive **Streamlit application** simulates the technical and ethical challenges of preparing biomedical datasets for collaborative AI research. It demonstrates how institutions can collaborate without compromising patient privacy using **Federated Learning** principles (inspired by NVIDIA FLARE).

## 📌 Features

* **Dual Tracks:** Choose between a **Clinical Track** (Hospital COVID-19 data) and a **Basic Science Track** (Immunology Lab data).
* **Interactive Data Generation:** Adjust sample sizes and outlier contamination rates to see how "dirty data" affects research.
* **Outlier Detection:** Visualize and remove anomalies using dynamic **Z-Score thresholds**.
* **Preprocessing Pipeline:** Apply **Imputation** (handling missing data) and **Standardization** (Min-Max/Z-Score scaling) to prepare data for AI.
* **Federated Learning Simulation:** Visualize how local model updates (weights) are aggregated into a global model **without** sharing raw patient rows.
* **System Monitoring:** Real-time tracking of CPU and RAM usage to simulate resource constraints.

## 📂 Project Structure

```text
├── app.py                # Main application source code
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

```

## 🚀 Installation & Setup

Follow these steps to run the app locally.

### 1. Prerequisites

Ensure you have **Python 3.8+** installed on your system.

### 2. Clone or Download

Download the `app.py` and `requirements.txt` files to a local folder.

### 3. Install Dependencies

Open your terminal/command prompt, navigate to the folder, and run:

```bash
pip install -r requirements.txt

```

### 4. Run the Application

Execute the following command to start the local server:

```bash
streamlit run app.py

```

A new tab should automatically open in your default web browser at `http://localhost:8501`.

## 📖 How to Use the App

The app guides you through a 4-step workflow:

1. **Local Data Inspection:**
* Select your "Track" (Clinical vs. Basic Science) in the sidebar.
* Review the raw data to identify "Institution" labels and potential errors.


2. **Outlier Detection:**
* Use the **Z-Score Threshold slider** to identify extreme values (measurement errors).
* *Action:* You must click **"Apply Filter and Proceed"** to clean the dataset.


3. **Preprocessing:**
* Choose an **Imputation Method** (e.g., Mean or Median) to fill missing values.
* Choose a **Scaler** (e.g., Min-Max) to normalize data ranges.


4. **Federated Learning Simulation:**
* Click **"Run Federated Learning Round"**.
* Observe how the "Global Model" is updated by averaging the weights from Institution A and Institution B, while the "Patient Records Shared" metric remains at **0**.



## 🛠️ Technologies Used

* **Streamlit:** For the interactive web interface.
* **Pandas & NumPy:** For data manipulation and simulation.
* **Scikit-Learn:** For imputation and scaling algorithms.
* **SciPy:** For Z-score statistical calculations.
* **Seaborn & Matplotlib:** For data visualization.

## 🔗 References

* **Concept:** Loftus, Tyler - "Navigating multi-institutional data sharing challenges"
* **Clinical Dataset Source:** [IC3 UF Public COVID-19 Dataset](https://ic3.center.ufl.edu/research/resources/datasets/)
* **Basic Science Dataset Source:** [ImmPort](https://www.immport.org/shared/home)
* **Federated Learning Tool:** [NVIDIA FLARE](https://nvidia.github.io/NVFlare/)

---

*Created for Module 3 Educational Demo.*
