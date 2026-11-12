.PHONY: install train api app test notebook clean

install:
	pip install -r requirements.txt

train:            ## data + model + calibration + figures + leakage study
	python train.py

api:
	uvicorn api.main:app --reload --port 8000

app:
	streamlit run app/streamlit_app.py

test:
	pytest -q

notebook:         ## rebuild and execute the case-study notebook (+ HTML export)
	python notebooks/build_notebook.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/Insurance_Claim_Risk_Case_Study.ipynb
	jupyter nbconvert --to html notebooks/Insurance_Claim_Risk_Case_Study.ipynb

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
