import os

class SaveData:

    def __init__(self, data, filename):
        self.data = data
        self.filename = filename

    def save_data(self):
        results_dir = "../resultados"
        os.makedirs(results_dir, exist_ok=True)

        file_path = os.path.join(results_dir, f"{self.filename}.txt")

        with open(file_path, "w") as txt_file:
            for key, value in self.data.items():
                txt_file.write(f"{key}: {value}\n")
        print(f"Datos guardados en {file_path}")