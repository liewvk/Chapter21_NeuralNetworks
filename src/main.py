import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    data_file = Path("data") / "student_results.csv"
    output_folder = Path("outputs")
    output_file = output_folder / "neural_network_results.csv"

    output_folder.mkdir(exist_ok=True)

    df = pd.read_csv(data_file)

    print("Student Result Dataset")
    print("----------------------")
    print(df)

    print()
    print("Result Counts")
    print("-------------")
    print(df["Result"].value_counts())

    X = df[["StudyHours", "Attendance", "AssignmentScore"]]
    y = df["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(8,),
        max_iter=2000,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    results = pd.DataFrame({
        "StudyHours": X_test["StudyHours"],
        "Attendance": X_test["Attendance"],
        "AssignmentScore": X_test["AssignmentScore"],
        "ActualResult": y_test,
        "PredictedResult": predictions
    })

    print()
    print("Prediction Results")
    print("------------------")
    print(results)

    print()
    print("Model Accuracy")
    print("--------------")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

    print()
    print("Classification Report")
    print("---------------------")
    print(classification_report(y_test, predictions))

    new_student = pd.DataFrame({
        "StudyHours": [6],
        "Attendance": [78],
        "AssignmentScore": [70]
    })

    new_student_scaled = scaler.transform(new_student)

    new_prediction = model.predict(new_student_scaled)
    probabilities = model.predict_proba(new_student_scaled)[0]

    print()
    print("New Student")
    print("-----------")
    print(new_student)

    print()
    print(f"Predicted result: {new_prediction[0]}")

    print()
    print("Prediction Probabilities")
    print("------------------------")

    for class_name, probability in zip(model.classes_, probabilities):
        print(f"{class_name}: {probability * 100:.2f}%")

    results.to_csv(output_file, index=False)

    print()
    print(f"Prediction results saved to: {output_file}")


main()
