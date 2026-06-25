# PixelWise: Out-of-Distribution Detection & HITL Infrastructure

## Project Overview

PixelWise is a web application featuring a classification backend equipped with a comparative Out-of-Distribution (OOD) detection pipeline and Human-in-the-Loop (HITL) infrastructure. The system addresses silent degradation caused by training-serving skew by identifying when real-world data diverges from the training distribution.

The architecture utilizes a Logistic Regression model as the primary classification backbone. To robustly detect anomalies, it employs a dual-method defense:

* **Output-based Detection:** A Maximum Softmax Probability (MSP) baseline.
* **Input-based Detection:** An Isolation Forest algorithm that isolates structural anomalies in the feature space without relying on decision boundaries.

To support continuous model refinement, flagged out-of-distribution inputs are not blocked but are instead logged with their raw pixel arrays and isolation scores into a PostgreSQL database, paving the way for a complete active-learning and HITL labeling pipeline.

---

## Technology Stack

* **Backend:** FastAPI, Python 3.10+
* **Machine Learning:** scikit-learn, numpy
* **Database:** PostgreSQL (managed via SQLAlchemy & psycopg2)
* **Deployment:** Nginx, Systemd (automated via shell scripts)

---

## Setup and Installation

The repository contains an automated provisioning script (`setup-server.sh`) that initializes the environment, trains the anomaly detector, and sets up the server components.

### 1. Prerequisites

Ensure your host machine has the necessary system packages installed:

```bash
sudo apt install -y python3-venv python3-pip postgresql postgresql-contrib nginx

```

### 2. Environment Configuration

Create a `.env` file in the root directory of the project based on the `.env.example` template. The application and deployment scripts require the following variables:

* **`SECRET_API_KEY`**: A secure key used for authenticating the `/classify` endpoint. The setup script injects this into the frontend.
* **`DEBUG`**: Toggles debug mode on or off. Ensure this is set to `false` in a production environment.
* **`MODEL_REPO`**: The Git URL pointing to the external repository containing the baseline Logistic Regression model artifacts.
* **`MODEL_VERSION`**: The specific branch, tag, or commit of the model repository to pull during setup.
* **`MODEL_PATH`**: The designated local file path where the downloaded digit classifier (`.pkl`) is stored.
* **`MODEL_PATH_ISO_FOREST`**: The local file path where the locally trained Isolation Forest model artifact is saved.
* **`DB_PASSWORD`**: The password assigned to the newly created `pixelwise` PostgreSQL database user.

### 3. Execution

Make the setup script executable and run it:

```bash
chmod +x setup-server.sh
./setup-server.sh

```

**What the script does:**

1. **Model Retrieval:** Clones the remote model repository specified in your `.env` and extracts the `.pkl` files and `MODELCARD.md` into a local `models/` directory.
2. **Environment Initialization:** Creates a Python virtual environment (`.venv`) and installs all dependencies from `requirements.txt`.
3. **OOD Pipeline Training:** Executes `train_isolation_forest.py` to fit the Isolation Forest unsupervised on the clean MNIST training set.
4. **Database Provisioning:** Creates the PostgreSQL `pixelwise` role and database.
5. **Production Deployment (If run as `produser`):**
* Deploys the frontend files to `/var/www/pixelwise`.
* Configures Nginx using `deploy/pixelwise.nginx` and reloads the web server.
* Installs and enables the `pixelwise` systemd service for the FastAPI backend.
* Sets up a systemd timer (`pixelwise-deploy.timer`) for automated background deployments.

---

## API Usage

The primary inference endpoint is accessible via `POST /classify`.

**Headers Required:**

* `x-api-key`: Must match the `SECRET_API_KEY` defined in your `.env`.

The endpoint accepts a JSON payload containing a batch of image pixels and returns the predicted digit, the confidence score, the individual class scores, and an `is_ood` boolean flag indicating whether the input breached the pre-calibrated TPR95 thresholds of either the MSP or the Isolation Forest models.