from sklearn.datasets import load_svmlight_file
import pandas as pd
from feature_competency import problem_feature_competency

# 1. Load LIBSVM file
X, y = load_svmlight_file("mushroom.txt")  # change filename

# 2. Convert sparse → dense
X = X.toarray()

# 3. Ensure binary labels (0/1)
# If labels are {1,2}, convert to {0,1}
if set(y) == {1.0, 2.0}:
    y = y - 1

# 4. Convert to DataFrame
X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y, columns=["label"])

col_to_comp = problem_feature_competency(X_df, y_df)
cols = col_to_comp['column'].astype(int).tolist()
print(cols)
X_df = X_df[cols]

# 5. Save to CSV
X_df.to_csv("questions.csv", index=False)
y_df.to_csv("solutions.csv", index=False)

print("Saved: questions.csv and solutions.csv")