import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score, accuracy_score

def draw_heatmap(X_test, y_test, model, label_encoder):

    predicted_probabilities = model.predict(x=X_test)
    predicted_index = np.argmax(predicted_probabilities, axis=1)

    y_test_index = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test_index, predicted_index)
    precision = precision_score(y_test_index, predicted_index, average='macro')
    recall = recall_score(y_test_index, predicted_index, average='macro')
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    cm = confusion_matrix(y_test_index, predicted_index)
    original_labels = np.unique(label_encoder.inverse_transform(np.argmax(y_test, axis=1)))
 
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=original_labels, yticklabels=original_labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    
    #plt.show()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()