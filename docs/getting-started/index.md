# Getting Started

## Installing pywaterflood

You can install pywaterflood with

```bash
pip install pywaterflood
```

## Running pywaterflood

First, import `pandas` and `pywaterflood`'s CRM module.

```python
import pandas as pd
from pywaterflood import CRM
```

Next, use pandas to read production and injection data.

```python
gh_url = (
    "https://raw.githubusercontent.com/frank1010111/pywaterflood/master/testing/data/"
)
prod = pd.read_csv(gh_url + "production.csv", header=None).values
inj = pd.read_csv(gh_url + "injection.csv", header=None).values
time = pd.read_csv(gh_url + "time.csv", header=None).values[:, 0]
```

Finally, run CRM and check the predictions and residuals.

```python
crm = CRM()
crm.fit(prod, inj, time)
q_hat = crm.predict()
residuals = crm.residual()
```

The connectivity matrix can be created thusly:

```python
connectivity = pd.DataFrame(crm.gains).rename_axis(index="Producer", columns="Injector")
connectivity.T.style.highlight_max().format("{:.2}")
```
