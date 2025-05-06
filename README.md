# COOLCAT: Entropy-Based Clustering for Categorical Data
> A **pure-Python**, dependency-light implementation of  
> COOLCAT (“Clustering Of Categorical data”) - Barbara et al., **PKDD 2002**

---

## Features

* **Works natively on categorical attributes** – no one-hot / embeddings needed.  
* **Entropy-minimisation principle** → crisp, interpretable clusters.  
* **No heavy deps**: NumPy & Pandas are *optional*; pure-Python fallback provided.  
* Fully typed, NumPy-style doc-strings, scikit-learn-like API (`fit`, `predict`, `fit_predict`).

  ```python
  from coolcat import Coolcat
  import pandas as pd
  
  df = pd.DataFrame({
      "color":  ["red","red","blue","green","blue","green","red","green"],
      "shape":  ["circle","square","circle","square","square","circle","circle","square"],
      "animal": ["cat","dog","cat","dog","dog","cat","dog","cat"],
  })
  
  model = Coolcat(
      n_clusters=3,
      batch_size=200,        # streaming buffer (optional)
      random_state=42,
  )
  
  labels = model.fit_predict(df)
  print(labels)   # e.g. [0, 0, 1, 2, 1, 2, 0, 2]
  ```

## Citing / references
Barbara D., Li Y., Couto J., Zhang J. (2002).
COOLCAT: An Entropy-Based Algorithm for Categorical Clustering.
In PKDD 2002 (pp. 64-78).
