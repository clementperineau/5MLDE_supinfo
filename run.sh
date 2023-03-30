apt-get update
apt-get -y upgrade
apt-get -y install zip unzip
unzip mlflow-titanic-v2.zip
pip install numpy pandas matplotlib==3.1.0 seaborn scikit-learn mlflow && \
    conda install -y -c conda-forge lightgbm xgboost
cd mlflow-titanic
python src/preprocess.py
python src/train.py
python src/predict.py
bash orchestration_01_prefect_orion.sh