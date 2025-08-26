# Requirement

conda env create -f audra_train_environment.yml

torch 1.8+

torchvision

Python 3

pip install ftfy regex tqdm

pip install git+https://github.com/openai/CLIP.git

# Run

1.get the AuDrA Drawings from: https://osf.io/h4adm/, add each of the folders to the directory with train_CSCA.py (e.g., primary_images folder)

2.run train_CSCA.py to train CSCA

3.run train_CSCA_LCR.py, train_CSCA_LCR_CCT.py, train_CSCA_LCR_SCT.py to train the model for the corresponding ablation experiment

4.run Visualization_1.py, Visualization_2.py to obtain corresponding visualization result of the paper.

5.run calculate_SRCC_PLCC.py to check the model's performance in different dataset

