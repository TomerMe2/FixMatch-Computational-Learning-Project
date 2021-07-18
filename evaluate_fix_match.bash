# This file contains the way that we used to evaluate FixMatch, our suggested improvement and the other well known algorithm.
# We used 1 epoch since we didn't have the computing power to do more epochs in feasible time

# To evaluate FixMatch:
python run_kfold_search.py DATASET.label_num=10 DATASET.strongaugment='RA' EXPERIMENT.epoch_n=1

# To evaluate our suggested improvement (Ratio Fix Match):
python run_kfold_search.py DATASET.label_num=10 DATASET.strongaugment='RA' EXPERIMENT.epoch_n=1 EXPERIMENT.is_suggested_improvement=1

# To evaluate label spreading (the model that we compare against)
python label_spreading_model.py

# To evaluate them using ANOVA and Tukey Post Hoc if needed
python compare_models.py