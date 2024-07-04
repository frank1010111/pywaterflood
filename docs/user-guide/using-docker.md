# Using Docker

If you want to run pywaterflood on, for instance, a supercomputer or maybe The Cloud™, the easiest way to deploy it is using Docker. In order to facilitate that, a dockerfile and docker image are provided as part of `pywaterflood`.

Imagine you have production, injection, and time csv files like those at <https://github.com/frank1010111/pywaterflood/tree/master/testing/data>, and they're in the `data` folder.

In Linux, you can copy test data with the script

```sh
mkdir data
cd data
for f in production.csv injection.csv time.csv; do
    curl -O https://raw.githubusercontent.com/frank1010111/pywaterflood/master/testing/data/$f
done
cd -
```

Next, create a `run_crm.py` file.

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
from pywaterflood import CRM

TAU_SELECTION = "per-pair"
CONSTRAINTS = "up-to one"
NUM_CORES = 8
data_dir = Path("/data")
prod = pd.read_csv(data_dir / "production.csv", header=None).to_numpy()
inj = pd.read_csv(data_dir / "injection.csv", header=None).to_numpy()
time = pd.read_csv(data_dir / "time.csv", header=None).to_numpy()[:, 0]

for run in range(10):
    crm = CRM(tau_selection=TAU_SELECTION, constraints=CONSTRAINTS)
    crm.fit(prod, inj, time, num_cores=NUM_CORES, random=True)
    crm.to_pickle(Path(__file__).parent / "results" / f"crm_results_{run+1}.pkl")
```

Then a dockerfile:

```
FROM frmale/pywaterflood:0.3.1
COPY --chown=pywaterflood:pywaterflood ./run_crm.py /home/pywaterflood/
WORKDIR /home/pywaterflood
RUN mkdir /home/pywaterflood/results && chown pywaterflood:pywaterflood /home/pywaterflood/results
ENTRYPOINT [ "python", "/home/pywaterflood/run_crm.py" ]
```

Create a volume to store the results:

```sh
docker volume create pywaterflood_results
```

Then, a docker compose file might look like this

```
services:
  pywaterflood:
    build: .
    volumes:
      - ./data:/data:ro
      - results:/home/pywaterflood/results:rw
volumes:
  results:
```

Alternatively, the docker command might be

```sh
docker run --rm -v ./data:/data:ro -v results:/home/pywaterflood/results:rw .
```

Extracting results from the docker volume can be done via docker-desktop or the one-liner

```sh
docker run --rm -v docker-pywaterflood_results:/src -v $(pwd)/results:/dest alpine sh -c 'cp -R /src/* /dest/'
```

This example, fully worked out, is available at <https://github.com/frank1010111/pywaterflood-containers>.
