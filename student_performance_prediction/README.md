# Student Performance Prediction System

This project demonstrates a simple machine learning pipeline to predict whether a student will pass or fail based on study hours, attendance rate, and previous exam scores.

## Project Structure

- `predict_student_performance.py` - Main Python script to load or generate the dataset, preprocess features, train a classifier, and evaluate predictions.
- `data/student_performance.csv` - Synthetic dataset generated automatically when the script runs if no file exists.
- `requirements.txt` - Python dependencies needed to run the project.

## Project Tasks

1. Create GitHub repository
   - Initialize a Git repository in the project folder.
   - Add project files and commit changes.
   - Push to GitHub and collaborate using branches and pull requests.

2. Load and explore dataset
   - The script loads `data/student_performance.csv` if it exists.
   - If the dataset does not exist, it generates a synthetic dataset and saves it.
   - Data exploration includes previewing rows, summary statistics, and target distribution.

3. Preprocess data
   - Convert the target label to numeric values.
   - Standardize numeric feature columns: study hours, attendance, and previous score.
   - Split data into training and test sets.

4. Train ML model
   - Train a `RandomForestClassifier` on the training data.
   - Evaluate model performance with accuracy, classification report, and confusion matrix.

5. Push code and collaborate
   - Use GitHub for version control.
   - Create branches for new features.
   - Open pull requests for review.

## Run the project

```bash
python student_performance_prediction/predict_student_performance.py
```

## Streamlit UI

Run the Streamlit app with:

```bash
streamlit run student_performance_prediction/streamlit_app.py
```

## Dataset

The project includes a real dataset at `student_performance_prediction/data/student_performance.csv`. The script will also regenerate the dataset automatically if the file is missing.

## GitHub repository

This project is pushed to GitHub:

https://github.com/vnmohan68-spec/student-performance-prediction-system

## CI workflow

A GitHub Actions workflow is configured at `.github/workflows/python-app.yml` to install dependencies and run the script on pushes and pull requests to `main`.

## Collaboration

Use branches for new features and create pull requests for review. Example:

```bash
git checkout -b feature/add-model-explanation
# make changes
git add .
git commit -m "Add model explanation"
git push origin feature/add-model-explanation
```

Then open a pull request on GitHub.
