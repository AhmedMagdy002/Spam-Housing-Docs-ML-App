import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Load the CSV data
df = pd.read_csv("bbc.csv", delimiter="\t")

# Extract the "content" and "category" columns
text_data = df['content']
categories = df['category']  # Assuming you have a 'category' column

# Initialize NLTK resources (if not already done)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the stemmer
stemmer = PorterStemmer()

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Initialize an empty list to store processed documents
processed_documents = []

# Text preprocessing loop
for document in text_data:
    # Convert text to lowercase
    document = document.lower()

    # Remove special characters and punctuation
    document = re.sub(r'[^a-zA-Z0-9\s]', '', document)

    # Tokenize the text into words
    words = word_tokenize(document)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    # Apply stemming to words
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    # Join the processed words back into a sentence
    processed_document = ' '.join(stemmed_words)

    # Append the processed document to the list
    processed_documents.append(processed_document)

# Apply TF-IDF vectorization to the processed documents
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_documents)

# Specify the number of clusters (in this case, 5)
n_clusters = 5

# Apply K-means clustering with 5 clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(tfidf_matrix)


df=pd.read_csv("emails.csv")
df = df.drop_duplicates()

# Your existing code for loading data and training models
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
    return text

df['cleaned_text'] = df['text'].apply(clean_text)
features = df['cleaned_text']
target = df['spam']

# Step 2: Preprocessing
vectorizer = CountVectorizer()  # You can also try TfidfVectorizer instead
X = vectorizer.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=1)

# Step 3: Develop the models
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
nb = MultinomialNB()

# Step 4: Training
knn.fit(X_train, y_train)
dt.fit(X_train, y_train)
nb.fit(X_train, y_train)

# Step 5: Testing
knn_predictions = knn.predict(X_test)
dt_predictions = dt.predict(X_test)
nb_predictions = nb.predict(X_test)

# Step 6: Evaluation
knn_accuracy = accuracy_score(y_test, knn_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)
nb_accuracy = accuracy_score(y_test, nb_predictions)

knn_cm = confusion_matrix(y_test, knn_predictions)
dt_cm = confusion_matrix(y_test, dt_predictions)
nb_cm = confusion_matrix(y_test, nb_predictions)



# Create GUI
class ModelEvaluationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Evaluation App")

        self.model_names = ["KNN", "Decision Tree", "Naive Bayes"]

        self.selected_models = []

        self.model_selection_frame = ttk.Frame(self.root)
        self.model_selection_frame.pack()

        self.accuracy_plot_frame = ttk.Frame(self.root)
        self.accuracy_plot_frame.pack()

        self.accuracy_plot_frame.rowconfigure(0, weight=1)
        self.accuracy_plot_frame.columnconfigure(0, weight=1)

        self.model_checkboxes = []
        for model_name in self.model_names:
            var = tk.IntVar()
            checkbox = ttk.Checkbutton(self.model_selection_frame, text=model_name, variable=var)
            checkbox.var = var
            self.model_checkboxes.append(checkbox)

        self.plot_button = ttk.Button(self.model_selection_frame, text="Plot Accuracy", command=self.plot_accuracy)
        self.plot_button.grid(row=len(self.model_names), columnspan=2, padx=10, pady=5)

        for i, checkbox in enumerate(self.model_checkboxes):
            checkbox.grid(row=i, column=0, sticky="w", padx=10, pady=2)

    def plot_accuracy(self):
        self.selected_models = [model_name for model_name, checkbox in zip(self.model_names, self.model_checkboxes) if checkbox.var.get() == 1]

        if not self.selected_models:
            return

        self.plot_accuracies(self.selected_models)

    def plot_accuracies(self, selected_models):
        
        selected_accuracies = []

        for model_name in selected_models:
            if model_name == "KNN":
                selected_accuracies.append(knn_accuracy)
            elif model_name == "Decision Tree":
                selected_accuracies.append(dt_accuracy)
            elif model_name == "Naive Bayes":
                selected_accuracies.append(nb_accuracy)

        plt.figure(figsize=(8, 6))
        plt.bar(selected_models, selected_accuracies, color='blue')
        plt.ylim(0, 1)
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison')

        for widget in self.accuracy_plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(plt.gcf(), master=self.accuracy_plot_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
# Import other necessary libraries here

# ... (Your housing model code) ...

# Load the housing data
data = pd.read_csv('Housing.csv')

# Data cleaning
data_cleaned = data.dropna()  # Remove rows with missing values

# Outlier removal using IQR method
def remove_outliers(df, col, threshold=3):
    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
    df = df[z_scores < threshold]
    return df

# Apply outlier removal to all features
data_removed_outliers = remove_outliers(data_cleaned, 'price')

# Separate the features (X) and target variable (y) after preprocessing
X = data_removed_outliers.drop('price', axis=1)
y = data_removed_outliers['price']

# Apply one-hot encoding to categorical features
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(X[categorical_features])
encoded_feature_names = encoder.get_feature_names(categorical_features)

# Create a DataFrame from the encoded features
data_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Combine the encoded features with numeric features
numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
scaler = MinMaxScaler()
data_numeric = pd.DataFrame(scaler.fit_transform(X[numeric_features]), columns=numeric_features)

data_processed = pd.concat([data_numeric, data_encoded], axis=1)

# Validate for invalid values (e.g., negative values) after preprocessing
invalid_values = data_processed[data_processed < 0].sum()
if invalid_values.any():
    print("Invalid values detected after preprocessing:\n", invalid_values)
else:
    print("No invalid values detected after preprocessing.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_processed, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2:.4f}")


class EmailModelEvaluation:
    def __init__(self, root, knn_accuracy, dt_accuracy, nb_accuracy, knn_cm, dt_cm, nb_cm):
        self.root = root
        self.root.title("Model Evaluation App")

        self.model_names = ["KNN", "Decision Tree", "Naive Bayes"]

        self.selected_models = []

        self.model_selection_frame = ttk.Frame(self.root)
        self.model_selection_frame.pack()

        self.accuracy_plot_frame = ttk.Frame(self.root)
        self.accuracy_plot_frame.pack()

        self.confusion_matrix_frame = ttk.Frame(self.root)
        self.confusion_matrix_frame.pack()

        self.accuracy_plot_frame.rowconfigure(0, weight=1)
        self.accuracy_plot_frame.columnconfigure(0, weight=1)

        self.confusion_matrix_frame.rowconfigure(0, weight=1)
        self.confusion_matrix_frame.columnconfigure(0, weight=1)

        self.model_checkboxes = []
        for model_name in self.model_names:
            var = tk.IntVar()
            checkbox = ttk.Checkbutton(self.model_selection_frame, text=model_name, variable=var)
            checkbox.var = var
            self.model_checkboxes.append(checkbox)

        self.plot_button = ttk.Button(self.model_selection_frame, text="Plot Accuracy", command=self.plot_accuracy)
        self.plot_button.grid(row=len(self.model_names), columnspan=2, padx=10, pady=5)

        self.confusion_matrix_button = ttk.Button(self.model_selection_frame, text="Show Confusion Matrix", command=self.show_confusion_matrix)
        self.confusion_matrix_button.grid(row=len(self.model_names) + 1, columnspan=2, padx=10, pady=5)

        for i, checkbox in enumerate(self.model_checkboxes):
            checkbox.grid(row=i, column=0, sticky="w", padx=10, pady=2)

        # Store confusion matrices for all models
        self.confusion_matrices = {}

        # Store accuracy values and confusion matrices
        self.knn_accuracy = knn_accuracy
        self.dt_accuracy = dt_accuracy
        self.nb_accuracy = nb_accuracy
        self.knn_cm = knn_cm
        self.dt_cm = dt_cm
        self.nb_cm = nb_cm

    def plot_accuracy(self):
        self.selected_models = [model_name for model_name, checkbox in zip(self.model_names, self.model_checkboxes) if checkbox.var.get() == 1]

        if not self.selected_models:
            return

        self.plot_accuracies(self.selected_models)

    def plot_accuracies(self, selected_models):
        selected_accuracies = []

        for model_name in selected_models:
            if model_name == "KNN":
                selected_accuracies.append(self.knn_accuracy)
            elif model_name == "Decision Tree":
                selected_accuracies.append(self.dt_accuracy)
            elif model_name == "Naive Bayes":
                selected_accuracies.append(self.nb_accuracy)

        plt.figure(figsize=(8, 6))
        plt.bar(selected_models, selected_accuracies, color='blue')
        plt.ylim(0, 1)
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison')

        for widget in self.accuracy_plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(plt.gcf(), master=self.accuracy_plot_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def show_confusion_matrix(self):
        if not self.selected_models:
            return

        for widget in self.confusion_matrix_frame.winfo_children():
            widget.destroy()

        # Clear the stored confusion matrices
        self.confusion_matrices.clear()

        # Calculate the number of columns for the grid layout
        num_columns = len(self.selected_models)

        # Create a single figure with multiple columns
        fig, axes = plt.subplots(1, num_columns, figsize=(6*num_columns, 6))
        fig.subplots_adjust(wspace=0.5)

        for col, model_name in enumerate(self.selected_models):
            if model_name == "KNN":
                cm = self.knn_cm
            elif model_name == "Decision Tree":
                cm = self.dt_cm
            elif model_name == "Naive Bayes":
                cm = self.nb_cm

            axes[col].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[col].set_title(f'{model_name} Confusion Matrix')
            axes[col].tick_params(axis='both', which='both', length=0)
            axes[col].set_xticks([0, 1])
            axes[col].set_yticks([0, 1])
            axes[col].set_xticklabels(['Not Spam', 'Spam'])
            axes[col].set_yticklabels(['Not Spam', 'Spam'])
            axes[col].set_xlabel('Predicted')
            axes[col].set_ylabel('True')

            # Store the confusion matrix for later use
            self.confusion_matrices[model_name] = cm

        canvas = FigureCanvasTkAgg(fig, master=self.confusion_matrix_frame)
        canvas.get_tk_widget().grid(row=0, column=0)
        canvas.draw()



class HousingModelEvaluationApp:
    def __init__(self, root, y_test, y_pred, r2):
        self.root = root
        self.root.title("Housing Model Evaluation")

        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack()

        self.plot_button = ttk.Button(self.plot_frame, text="Show Scatter Plot", command=self.show_scatter_plot)
        self.plot_button.pack(padx=10, pady=5)

        self.y_test = y_test
        self.y_pred = y_pred
        self.r2 = r2

    def show_scatter_plot(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, self.y_pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values")

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(plt.gcf(), master=self.plot_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

        r2_label = ttk.Label(self.plot_frame, text=f"R^2 Score: {self.r2:.4f}", font=("Arial", 14))
        r2_label.pack(padx=10, pady=5)

class DocumentClassificationApp:
    def __init__(self, root, tfidf_matrix, kmeans, categories):
        self.root = root
        self.root.title("Document Classification Visualization")

        # Reduce the TF-IDF matrix to 2D using PCA for visualization
        pca = PCA(n_components=2)
        tfidf_matrix_2d = pca.fit_transform(tfidf_matrix.toarray())

        # Map cluster numbers to actual category labels
        cluster_labels = kmeans.labels_
        label_mapping = {cluster_num: actual_label for cluster_num, actual_label in zip(range(len(categories)), categories.unique())}
        actual_labels = [label_mapping[cluster_num] for cluster_num in cluster_labels]

        # Ensure labels match the length of data used for clustering
        actual_labels = actual_labels[:tfidf_matrix_2d.shape[0]]

        # Add the actual labels to the DataFrame
        df['actual_label'] = actual_labels

        # Create a scatter plot for visualization
        plt.figure(figsize=(10, 6))
        unique_labels = df['actual_label'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))  # Use a colormap for colors

        for label, color in zip(unique_labels, colors):
            cluster_mask = df['actual_label'] == label
            plt.scatter(
                tfidf_matrix_2d[cluster_mask, 0],
                tfidf_matrix_2d[cluster_mask, 1],
                label=label,
                color=color,
            )

        plt.legend()
        plt.title('Cluster Visualization Using PCA (Actual Labels)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # Create a Tkinter window and display the visualization
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack()
        self.figure = plt.gcf()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

class ModelSelectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Selection")

        model_type_label = ttk.Label(self.root, text="Select a Model:")
        model_type_label.pack(padx=10, pady=5)

        self.model_type = tk.StringVar()

        model_type_menu = ttk.OptionMenu(self.root, self.model_type, "Select Dataset", "Emails", "Housing", "Document Classification")
        model_type_menu.pack(padx=10, pady=5)

        start_button = ttk.Button(self.root, text="Start Evaluation", command=self.open_evaluation_window)
        start_button.pack(padx=10, pady=10)

    def open_evaluation_window(self):
        selected_option = self.model_type.get()
        if selected_option == "Emails":
            email_window = tk.Toplevel(self.root)
            email_window.title("Email Model Evaluation")
            email_app = EmailModelEvaluation(email_window, knn_accuracy, dt_accuracy, nb_accuracy, knn_cm, dt_cm, nb_cm)
        elif selected_option == "Housing":
            housing_window = tk.Toplevel(self.root)
            housing_window.title("Housing Model Evaluation")
            housing_app = HousingModelEvaluationApp(housing_window, y_test, y_pred, r2)
        elif selected_option == "Document Classification":
            document_classification_window = tk.Toplevel(self.root)
            document_classification_window.title("Document Classification Visualization")
            document_classification_app = DocumentClassificationApp(document_classification_window, tfidf_matrix, kmeans, categories)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelSelectionApp(root)
    root.mainloop()
