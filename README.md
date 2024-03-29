social_capital_in_trade_networks
==============================

This repo contains the code for the study

Data Sources 
------------
The following raw datasets are use in this work: 
- ICIO2018_20**.zip: Inter-Country Input Output (ICIO) tables - http://www.oecd.org/sti/ind/inter-country-input-output-tables.htm
- MIG_12082020131505678.csv : International migration database - https://stats.oecd.org/Index.aspx?DataSetCode=MIG
- DP_LIVE_13102020161705689.csv: Gini coefficient - https://data.oecd.org/inequality/income-inequality.htm
- DP_LIVE_06072020200357239.csv: Population - https://data.oecd.org/pop/population.htm
- DP_LIVE_06072020184943320.csv: Working population expressed as a percentage of the country's total population - https://data.oecd.org/pop/working-age-population.
- API_NE.GDI.TOTL.CD_DS2_en_excel_v2_1742937.xls: Gross capital formation (current US$) from World Bank database - https://data.worldbank.org/indicator/NE.GDI.TOTL.CD?page=1 

htm#indicator-chart


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
