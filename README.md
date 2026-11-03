
# Air Pollution Prediction System

An intelligent air pollution prediction system that combines Machine Learning and MongoDB to analyze environmental data, predict air quality levels, and support data-driven environmental monitoring.

## Features

- Machine Learning-based air pollution prediction.
- Analysis of historical air quality data.
- Interactive Streamlit dashboard.
- MongoDB integration for storing and managing environmental data.
- Data visualization and pollution trend analysis.

## Technologies Used

- Python
- Machine Learning (Scikit-learn)
- Streamlit
- MongoDB
- MongoDB Compass
- Pandas, NumPy, Plotly

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/aaralyngeorge05/Air-Pollution-Prediction-System.git
cd Air-Pollution-Prediction-System
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows:**

```powershell
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Set up MongoDB

Install MongoDB Community Server and ensure the MongoDB service is running.

MongoDB Compass can be used to view and manage the database.

The application currently uses a local MongoDB connection.

### 6. Obtain the trained ML model

Download `air_quality_model.pkl` separately and place it in the project root folder, alongside `app.py`.

### 7. Run the application

```bash
streamlit run app.py
```

The application will open in your default browser.

## Project Structure

```text
Air-Pollution-Prediction-System/
│
├── app.py
├── app_backup.py
├── import_history.py
├── city_day.csv
├── air_quality_model.pkl
├── requirements.txt
├── image.png
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Author

**Aaralyn George**

GitHub: [@aaralyngeorge05](https://github.com/aaralyngeorge05)
