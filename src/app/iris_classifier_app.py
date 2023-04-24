import re
import tkinter as tk
from tkinter import messagebox
from knn.data_loader import DataLoader
from knn.preprocessor import Preprocessor
from knn.cross_validator import CrossValidator
from knn.k_nearest_neighbors import KNearestNeighbors
from knn.iris_model import IrisModel

class IrisClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris Flower Classifier")
        self.model = IrisModel("data/iris.txt", num_neighbors=10, n_folds=5, seed_value=2)
        self.create_widgets()

        # Bind the Escape key to close the app
        self.root.bind('<Escape>', lambda e: self.root.destroy())

        # Predefined test cases
        self.test_cases = [
            {"values": [5.1, 3.8, 1.6, 0.2], "species": "Iris-setosa"},
            {"values": [5, 2, 3.5, 1.0], "species": "Iris-versicolor"},
            {"values": [7.9, 3.8, 6.4, 2.0], "species": "Iris-virginica"},
        ]
        self.current_case_index = 0
        self.populate_entries(self.test_cases[self.current_case_index]["values"])

        # Allow the window to resize based on the content
        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_width(), self.root.winfo_height())

        # Display the cross-validation results
        self.display_cross_validation_results()

    def display_cross_validation_results(self):
        mean_accuracy, formatted_scores = self.model.get_scores()
        self.results_text.set(f"Mean Accuracy: {mean_accuracy}\nAccuracy per fold: {formatted_scores}")

    def create_widgets(self):
        # Set font for labels and entries
        label_font = ("Arial", 16, "bold")
        entry_font = ("Arial", 16)
        results_font = ("Arial", 14)
        button_font = ("Arial", 16, "bold")

        # Create a menu bar
        menu_font = ("Arial", 12)  # Font for the menu bar

        menu_bar = tk.Menu(self.root, font=menu_font)

        # Add a 'File' menu
        file_menu = tk.Menu(menu_bar, tearoff=0, font=menu_font)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        # Add a 'Help' menu
        help_menu = tk.Menu(menu_bar, tearoff=0, font=menu_font)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        # Configure the menu bar
        self.root.config(menu=menu_bar)

        # Input labels and entries
        labels = ["Sepal Length:", "Sepal Width:", "Petal Length:", "Petal Width:"]
        self.entries = []

        for i, label_text in enumerate(labels):
            # Create and place the label
            label = tk.Label(self.root, text=label_text, font=label_font)
            label.grid(row=i, column=0, pady=5, padx=0, sticky="e")

            # Create and place the entry box
            entry = tk.Entry(self.root, font=entry_font, width=20)
            entry.grid(row=i, column=1, pady=5, padx=5, sticky="w")
            self.entries.append(entry)

        # Cross-validation results label
        cv_label = tk.Label(self.root, text="Cross-Validation Results:", font=label_font)
        cv_label.grid(row=4, columnspan=2, pady=(15, 5))

        # Results display
        self.results_text = tk.StringVar()
        self.results_label = tk.Label(self.root, textvariable=self.results_text, wraplength=300, font=results_font)
        self.results_label.grid(row=5, columnspan=2, pady=10)

        # Buttons in the same row, side by side
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=6, columnspan=2, pady=10)

        # Generate button
        self.generate_button = tk.Button(button_frame, text="Generate", font=button_font, command=self.generate_case)
        self.generate_button.pack(side="left", padx=10)

        # Predict button
        self.predict_button = tk.Button(button_frame, text="Predict", font=button_font, command=self.predict)
        self.predict_button.pack(side="right", padx=10)


    def populate_entries(self, values):
        for entry, value in zip(self.entries, values):
            entry.delete(0, tk.END)
            entry.insert(0, str(value))

    def generate_case(self):
        self.current_case_index = (self.current_case_index + 1) % len(self.test_cases)
        case = self.test_cases[self.current_case_index]
        self.populate_entries(case["values"])
        # messagebox.showinfo("Test Case", f"Generated Test Case: {case['species']}")

    def predict(self):
        try:
            # Get inputs and make a prediction
            inputs = [float(entry.get()) for entry in self.entries]
            if len(inputs) !=4:
                raise ValueError

            prediction = self.model.predict(inputs)

            messagebox.showinfo("Prediction", f"Predicted Iris Species: {prediction}")

        except ValueError:
            messagebox.showerror("Input Error", "Please enter four valid floating point numbers.")

    def show_about(self):
        messagebox.showinfo("About", "Iris Flower Classifier\nVersion 1.0")

def main():
    root = tk.Tk()
    app = IrisClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()