# UF team in 2019 N2C2 challenge Track2 Family history extraction from clinical narratives

## challenge link
- https://n2c2.dbmi.hms.harvard.edu/track2

## Aim
- In this challenge, we developed a end-to-end system for FH extraction. Our best system consisted of a BERT NER model, four BERT classification model, and a BERT relation identification model.  We also adopted a majority voting strategy for concept extraction ensemble.

## Dependencies
- We adopted the BERT NER model implementation from https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER.git
- we did not include the package in this repo but you can download it either using git command or see setup section

## setup
- we have all models and dependencies available for download
- you can run ```. shell/setup.sh``` to download all necessary models

## data
- we have the doc_test_1055.txt as the sample test from the challenge test dataset.
- we did not include text preprocessing (tokenization) here. You can develop you own method.
- if you need the whole dataset, please contact the 2019 n2c2 challenge organizer.

## project description
- we have sample preprocessed data (partial challenge data) in the sample_data dir for each subtasks in test
- we have all the fine-tuned models available for download through Amazon AWS S3
- You can run system on the sample data using ```python app.py```, the results will be generated in ./results/pred
- we have the gold standard for the sample data in ./results/gs
- Performances will be printed in console
- The system here is the best one we reported in our publication.

## models
- see ./shell/setup.sh for details

## Authors
- Xi Yang (alexgre@ufl.edu)
- Xing He (hexing@ufl.edu)
- Hansi Zhang (hansi.zhang@ufl.edu)
- Jiang Bian (bianjiang@ufl.edu)
- Yonghui Wu (yonghui.wu@ufl.edu)

## cite our work
please cite our work as
