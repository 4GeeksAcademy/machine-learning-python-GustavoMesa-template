from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "reviews.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "reviews_with_sentiment.csv"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_LABELS = ["Negativa", "Neutral", "Positiva"]


def load_reviews(input_path=DEFAULT_INPUT):
	reviews = pd.read_csv(input_path)
	required_columns = {"review_id", "rating", "review_text"}
	missing_columns = required_columns.difference(reviews.columns)
	if missing_columns:
		raise ValueError(f"Faltan columnas requeridas: {sorted(missing_columns)}")
	if reviews["review_text"].isna().any():
		raise ValueError("La columna review_text contiene valores vacíos.")

	ratings = pd.to_numeric(reviews["rating"], errors="raise")
	if not ratings.isin([1, 2, 3, 4, 5]).all():
		raise ValueError("rating debe contener únicamente valores entre 1 y 5.")

	return reviews


def add_sentiment_predictions(reviews, model_name=MODEL_NAME, batch_size=16):
	from transformers import pipeline

	results = reviews.copy()
	results["sentiment_from_rating"] = results["rating"].map(
		{1: "Negativa", 2: "Negativa", 3: "Neutral", 4: "Positiva", 5: "Positiva"}
	)

	classifier = pipeline("sentiment-analysis", model=model_name, device=-1)
	predictions = classifier(
		results["review_text"].astype(str).tolist(),
		batch_size=batch_size,
		truncation=True,
	)
	label_mapping = {"NEGATIVE": "Negativa", "POSITIVE": "Positiva"}
	model_labels = [prediction["label"].upper() for prediction in predictions]
	unsupported_labels = set(model_labels).difference(label_mapping)
	if unsupported_labels:
		raise ValueError(f"Etiquetas inesperadas del modelo: {sorted(unsupported_labels)}")

	results["sentiment_model"] = [label_mapping[label] for label in model_labels]
	results["sentiment_score"] = [float(prediction["score"]) for prediction in predictions]
	return results


def evaluate_predictions(results):
	from sklearn.metrics import classification_report, confusion_matrix

	rating_labels = results["sentiment_from_rating"]
	model_labels = results["sentiment_model"]
	binary_mask = rating_labels != "Neutral"
	binary_rating_labels = rating_labels[binary_mask]
	binary_model_labels = model_labels[binary_mask]

	return {
		"three_class_report": classification_report(
			rating_labels,
			model_labels,
			labels=SENTIMENT_LABELS,
			output_dict=True,
			zero_division=0,
		),
		"three_class_confusion": confusion_matrix(
			rating_labels, model_labels, labels=SENTIMENT_LABELS
		),
		"binary_report": classification_report(
			binary_rating_labels,
			binary_model_labels,
			labels=["Negativa", "Positiva"],
			output_dict=True,
			zero_division=0,
		),
		"binary_confusion": confusion_matrix(
			binary_rating_labels,
			binary_model_labels,
			labels=["Negativa", "Positiva"],
		),
		"false_positives": results.loc[
			binary_mask & (rating_labels == "Negativa") & (model_labels == "Positiva")
		],
		"false_negatives": results.loc[
			binary_mask & (rating_labels == "Positiva") & (model_labels == "Negativa")
		],
	}


def main():
	reviews = load_reviews()
	results = add_sentiment_predictions(reviews)
	results.to_csv(DEFAULT_OUTPUT, index=False)
	evaluation = evaluate_predictions(results)

	print(f"Reseñas procesadas: {len(results)}")
	print(f"CSV guardado en: {DEFAULT_OUTPUT}")
	print("\nEvaluación de tres clases (incluye las reseñas de 3 estrellas):")
	print(pd.DataFrame(evaluation["three_class_report"]).transpose().round(3))
	print("\nEvaluación binaria (excluye las reseñas de 3 estrellas):")
	print(pd.DataFrame(evaluation["binary_report"]).transpose().round(3))
	print(f"\nFalsos positivos: {len(evaluation['false_positives'])}")
	print(f"Falsos negativos: {len(evaluation['false_negatives'])}")


if __name__ == "__main__":
	main()
