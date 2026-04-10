import pandas as pd
import numpy as np

def problem_feature_competency(qs: pd.DataFrame, sols: pd.DataFrame):
    Q = qs.values
    S = sols.values
    
    imp_1s = (Q.T @ S).astype(np.float32) # q == 1 and s == 1
    imp_2s = ((1 - Q).T @ S).astype(np.float32) # q == 0 and s == 1
    
    q_1_freq = Q.sum(axis=0).astype(np.float32)
    q_2_freq = (1 - Q).sum(axis=0).astype(np.float32)
    
    imp_1s = imp_1s / (q_1_freq[:, None] + 1e-6)
    imp_2s = imp_2s / (q_2_freq[:, None] + 1e-6)
    
    p_q1 = q_1_freq / len(Q)
    p_q0 = q_2_freq / len(Q)
    comp_vecs = np.abs(
        p_q1[:, None] * imp_1s -
        p_q0[:, None] * imp_2s
    )
    
    perfection = np.array([1] * len(sols.columns)).reshape(1, -1)
    competency = 1 - (np.linalg.norm(comp_vecs - perfection, axis=1) / np.sqrt(len(sols.columns)))
    
    return pd.DataFrame(
        {
            "column": qs.columns,
            "competency": competency
        }
    ).sort_values("competency", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    qs = pd.read_csv('questions.csv')
    sols = pd.read_csv('solutions.csv')
    print(problem_feature_competency(qs, sols))