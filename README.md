# Time Series Analytics for Page View Number

## Preparation

```
pipenv install
```

```
cat data/raw/xxx.export.txt | grep 'DATE' > data/processed/entry_date.txt
python src/make_dataset.py
```
