#!/bin/bash

if [ "$(echo "$(pwd)" | rev | cut -d'/' -f1 | rev)" != '2019_N2C2_Track2_FHextraction' ]
then
  echo "You have to change directory to 2019_N2C2_Track2_FHextraction"
  exit 1
fi

# set up NER module ClinicalTransformerNER
git clone https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER.git

mkdir ./models
cd  ./models || exit 1

# download NER models
for i in 0 1 2 3 4
do
  rm -f 2019_n2c2_fh_ner_${i}.zip
  wget https://transformer-models.s3.amazonaws.com/2019_n2c2_fh_ner_${i}.zip
  unzip 2019_n2c2_fh_ner_${i}.zip
#  mv 2019_n2c2_fh_ner_${i} ../models/2019_n2c2_fh_ner_${i}
  rm -f 2019_n2c2_fh_ner_${i}.zip
done

#download classification and relation models
tags=('obn' 'fmr' 'fms' 'lss' 'rel')
for i in ${tags[*]}
do
  rm -f 2019_n2c2_fh_"${i}".zip
  wget https://transformer-models.s3.amazonaws.com/2019_n2c2_fh_"${i}".zip
  unzip 2019_n2c2_fh_"${i}".zip
#  mv 2019_n2c2_fh_ner_"${i}" ../models/2019_n2c2_fh_ner_"${i}"
  rm -f 2019_n2c2_fh_"${i}".zip
done

cd ..

echo "set up done"