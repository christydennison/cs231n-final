mkdir ~/cs231n-final/dataset
cd ~/cs231n-final/dataset
kg download -u cs231n2017 -p cs231n2017 -c invasive-species-monitoring

mkdir ~/cs231n-final/dataset/train
cd ~/cs231n-final/dataset/train
7za e ../train.7z

mkdir ~/cs231n-final/dataset/test
cd ~/cs231n-final/dataset/test
7za e ../test.7z

mkdir ~/cs231n-final/dataset/labels
cd ~/cs231n-final/dataset/labels
unzip ../train_labels.csv.zip

mkdir ~/cs231n-final/dataset/submission
cd ~/cs231n-final/dataset/submission
unzip ../sample_submission.csv.zip

