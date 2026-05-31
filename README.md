Requirements_conda.txt can be used to create a conda environment to run all the files -> this is how it was done in production.
Requirements.txt can be used for pip install -r requirements.txt if this is the prefered style.

train_new.py, test_new.py and stat_test.py are the main working scripts used to collect data for the thesis. all trained models are provided within the root directory. 
test.py and train.py in outdated_but_working produced acceptable results but lack temporal understanding within the models leading to worse results. all trained models that have been tested for with this version as well as all results are provided.

all helper scripts have been either used for data annotation/pre processing or to create plots and tables to be used in the thesis.

to run the project simply create a conda environment using the provided requirements and run the test_new.py and stat_test.py file with your desired configurations in the scripts for past_len and future_len. If there is no model for your desired combination you can train new models using train_new.py after you changed your past_len and future_len in it-
