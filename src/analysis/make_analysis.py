# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import papermill as pm

@click.command()
@click.argument("notebooks_filepath", type=click.Path(exists=True))
@click.argument("data_filepath", type=click.Path())
def main(notebooks_filepath, data_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Executing all analysis notebooks")

    notebook_list = [
        "01_social_capital_regression_analysis.ipynb", 
        "02_PowerlawDistribution.ipynb",
        "03_tSNE_representations.ipynb",
        "04_NetworkDescripiton.ipynb",
        "05_NetworkEfficiency.ipynb",
        "06_ECI_correlation.ipynb",
        "07_diversity_orthogonality.ipynb",
        "08_dynamic_range_centralities.ipynb"                            
    ]
    for n in notebook_list:

        pm.execute_notebook(os.path.join(notebooks_filepath, n), 
                            os.path.join(notebooks_filepath, 'runs', n), 
                            parameters=dict(output_filepath=data_filepath)
                            )

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
