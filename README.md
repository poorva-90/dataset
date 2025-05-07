# Enron Email Analysis

This project analyzes the Enron email dataset using Python.

## Dataset

Download using:

```python
from kagglehub import dataset_download
path = dataset_download("wcukierski/enron-email-dataset")
df = pd.read_csv(path / "emails.csv")
