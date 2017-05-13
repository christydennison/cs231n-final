mkdir ~/final/dataset
cd ~/final/dataset
kg download -u cs231n2017 -p cs231n2017 -c invasive-species-monitoring

mkdir ~/final/dataset/train
cd ~/final/dataset/train
7za e ../train.7z

mkdir ~/final/dataset/test
cd ~/final/dataset/test
7za e ../test.7z

mkdir ~/final/dataset/labels
cd ~/final/dataset/labels
unzip ../train_labels.csv.zip

mkdir ~/final/dataset/submission
cd ~/final/dataset/submission
unzip ../sample_submission.csv.zip

