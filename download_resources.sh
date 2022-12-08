#!/usr/bin/env bash
mkdir -p resources
echo downloading SemEval 2013 task 13 data and evaluation code...
wget https://www.cs.york.ac.uk/semeval-2013/task13/data/uploads/semeval-2013-task-13-test-data.zip -P resources

# unzip might not be installed
python - <<EOF
import zipfile
with zipfile.ZipFile("./resources/semeval-2013-task-13-test-data.zip","r") as zip_ref:
    zip_ref.extractall('resources')
EOF

echo downloading SemEval 2010 task 14 data and evaluation code...
wget https://www.cs.york.ac.uk/semeval2010_WSI/files/evaluation.zip  -O resources/se2010eval.zip
wget https://www.cs.york.ac.uk/semeval2010_WSI/files/test_data.tar.gz  -O resources/se2010test_data.tar.gz
mkdir -p resources/SemEval-2010
python - <<EOF
import zipfile
with zipfile.ZipFile("./resources/se2010eval.zip","r") as zip_ref:
    zip_ref.extractall('./resources/SemEval-2010')
EOF

tar -C resources/SemEval-2010 -xzf resources/se2010test_data.tar.gz

echo downloading BART generated definitions...
wget https://drive.google.com/uc?export=download&id=1R5Pvevo9MW6mO2yymb8dvew_Z18VUxYT -O resources/bart_wsi_test_evaluation_n10_reranking.csv
