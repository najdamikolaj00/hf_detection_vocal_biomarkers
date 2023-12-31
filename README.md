# Heart failure detection based on vocal biomarkers

Our objective is to examine voice samples gathered from individuals affected with heart failure (HF) and leverage machine learning (ML) methodologies to investigate the relationship between variations in vocal attributes and shifts in the health condition of HF patients. 

## Research article

If accepted,

## Data access

Data are protected by privacy and GDPR regulations.

### Try out the code

```
# Clone the repo
git clone

# Create a virtual environment
python3 -m virtualenv .venv
source .venv/bin/activate

# Install the requirements
pip install -r requirements.txt
```
### Data file

- Structure of data file:

| column name     | dtype        | description
| -------------   | ----------   | -----------
| id              | str or int   | unique identifier of each subject
| class           | 0 or 1       | target of classification
| feature_1       | float        | feature extracted via OpenSmile
| feature_2       | float        | feature extracted via OpenSmile
| feature_n       | float        | feature extracted via OpenSmile
where n is equal to 6503

## License

This project is licensed under the terms of the MIT license.

## Acknowledgement

We extend our heartfelt gratitude to all who participated in our study, as your generosity and dedication are invaluable contributions to advancing medical knowledge and understanding of conditions such as heart failure.

## Funding

The study was funded by Miniatura 6 Grant, no 2022/06/X/ST6/01191, funded by the National Science Centre in Poland, and by Statutory funds of the Medical University of Silesia in Poland (no. (PCN-1-005/N/0/K and PCN-1-139/N/2/K).

