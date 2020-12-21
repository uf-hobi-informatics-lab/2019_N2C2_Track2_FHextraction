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
- we did not include any data in the current repo due to DUA. If you need our sample data or the whole dataset, please contact the 2019 n2c2 challenge organizer.

## project description
- we have all the fine-tuned models available for download through Amazon AWS S3
- You can run system on the sample data using ```python app.py```
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
please cite our work:

- https://medinform.jmir.org/2020/12/e22982
```
Yang X, Zhang H, He X, Bian J, Wu Y
Extracting Family History of Patients From Clinical Narratives: Exploring an End-to-End Solution With Deep Learning Models
JMIR Med Inform 2020;8(12):e22982
DOI: 10.2196/22982
PMID: 33320104
```
