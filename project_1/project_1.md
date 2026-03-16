1. Download the dataset using
```bash
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
```
2. Unzip the downloaded files under /data
3. Required packages:
```bash
pip install -r requirements.txt
```
4. Prepare the training set and the validation set with
```bash
cd src
python prepare_dataset.py
```
5. Train the models for deblurring and denoising using
```bash
mkdir models
cd src
python train.py
```
6. Test the models and store the result csv in /results using
```bash
mkdir results
cd src
python evaluate.py
```