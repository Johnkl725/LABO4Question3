import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

class KMeansClusteringApp:
    def __init__(self, master):
        self.master = master
        master.title("Clustering con K-Means")
        master.geometry("520x480")

        self.dataset = None
        self.numeric_columns = []

        # Botón para cargar el dataset
        self.load_button = tk.Button(master, text="Cargar Dataset CSV", command=self.load_dataset)
        self.load_button.pack(pady=10)

        # Lista de variables numéricas
        self.columns_label = tk.Label(master, text="Seleccione variables numéricas:")
        self.columns_label.pack()
        self.columns_listbox = tk.Listbox(master, selectmode='multiple', width=60)
        self.columns_listbox.pack(pady=5)

        # Entrada para el valor de K
        self.k_label = tk.Label(master, text="Ingrese el valor de K (2-20):")
        self.k_label.pack(pady=5)
        self.k_entry = tk.Entry(master)
        self.k_entry.pack(pady=5)

        # Botón para ejecutar K-Means
        self.run_button = tk.Button(master, text="Ejecutar K-Means", command=self.run_kmeans)
        self.run_button.pack(pady=10)

        # Botón para mostrar gráfico de codo
        self.elbow_button = tk.Button(master, text="Mostrar gráfico de codo", command=self.plot_elbow)
        self.elbow_button.pack(pady=5)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.dataset = pd.read_csv(file_path)
                self.numeric_columns = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
                self.columns_listbox.delete(0, tk.END)
                for col in self.numeric_columns:
                    self.columns_listbox.insert(tk.END, col)
                messagebox.showinfo("Éxito", "Dataset cargado correctamente.")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el dataset:\n{e}")

    def run_kmeans(self):
        if self.dataset is None:
            messagebox.showwarning("Advertencia", "Por favor, cargue un dataset primero.")
            return

        selected_indices = self.columns_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Advertencia", "Seleccione al menos una variable numérica.")
            return

        selected_columns = [self.numeric_columns[i] for i in selected_indices]
        data = self.dataset[selected_columns].dropna()

        try:
            k = int(self.k_entry.get())
            if k < 2 or k > 20:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Ingrese un valor válido para K (entre 2 y 20).")
            return

        # Escalado de datos
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Aplicar K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(data_scaled)
        sil_score = silhouette_score(data_scaled, labels)

        # Agregar etiquetas al DataFrame
        clustered_data = data.copy()
        clustered_data["Cluster"] = labels

        # Resumen estadístico por clúster
        summary = clustered_data.groupby("Cluster").mean().round(2)
        print("\nResumen por clúster:\n", summary)

        # Reducción de dimensionalidad para visualización
        pca = PCA(n_components=2)
        components = pca.fit_transform(data_scaled)

        # Visualización PCA
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=labels, palette="Set2", legend="full")
        plt.title(f"Clustering con K={k} | Silhouette Score = {sil_score:.3f}")
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.tight_layout()
        plt.show()

        # Conteo de puntos por clúster
        plt.figure(figsize=(6, 4))
        sns.countplot(x=labels)
        plt.title("Cantidad de puntos por clúster")
        plt.xlabel("Clúster")
        plt.ylabel("Cantidad")
        plt.tight_layout()
        plt.show()

        # Gráfico de Silhouette por muestra
        silhouette_vals = silhouette_samples(data_scaled, labels)
        plt.figure(figsize=(8, 5))
        y_lower = 10
        for i in range(k):
            ith_cluster_vals = silhouette_vals[labels == i]
            ith_cluster_vals.sort()
            size_cluster_i = ith_cluster_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_vals)
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        plt.axvline(x=sil_score, color="red", linestyle="--")
        plt.xlabel("Coeficiente de Silhouette")
        plt.ylabel("Clúster")
        plt.title("Gráfico de Silhouette por muestra")
        plt.tight_layout()
        plt.show()

        messagebox.showinfo("Resultado", f"Silhouette Score para K={k}: {sil_score:.3f}\n\nResumen por clúster impreso en consola.")

    def plot_elbow(self):
        if self.dataset is None:
            messagebox.showwarning("Advertencia", "Primero cargue un dataset.")
            return

        selected_indices = self.columns_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Advertencia", "Seleccione al menos una variable numérica.")
            return

        selected_columns = [self.numeric_columns[i] for i in selected_indices]
        data = self.dataset[selected_columns].dropna()

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        inertias = []
        ks = range(2, 21)
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init='auto')
            km.fit(data_scaled)
            inertias.append(km.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(ks, inertias, marker='o')
        plt.xlabel("Número de Clústers (K)")
        plt.ylabel("Inercia (Suma de Distancias)")
        plt.title("Método del Codo para Selección de K")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = KMeansClusteringApp(root)
    root.mainloop()