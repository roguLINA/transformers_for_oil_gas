Similarity learning via Transformers: representing time series from oil\&gas
=====

Code for experiments from the article of the same name. Include framework `trans_oil_gas` for dataset of well-intervals generation and training and testing Transformer-based (with Transformer, Informer, and Performer) Siamese and Triplet models. 

Installation of `trans_oil_gas`:
-----
1. Clone this repository
2. Install all necessary libraries via command in terminal: `pip install transformers_for_oil_gas/`
3. Use our framework via importing modules with names started with `utils_*` from `trans_oil_gas` 

Reproducing experiments from the article
-----
To reproduce all our experiments from the article "Similarity learning via Transformers: representing time series from oil\&gas":
1. Open `notebooks` folder.
2. Run jupyter notebook `all_models.ipynb`. It will train all models (Siamese and Triplet Transformer, Informer, and Performer).
3. Run experiments in other notebooks. It will use the pretrained models obtained in step 2. 

License
-----
The project is distributed under [MIT License](https://github.com/roguLINA/transformers_for_oil_gas/blob/main/License.txt).
