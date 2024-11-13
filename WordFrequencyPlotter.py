import matplotlib.pyplot as plt

class WordFrequencyPlotter:
    def plot_top_20_words(self, top_20_words_df):
        # Convertir el DataFrame de Spark a una lista de Python para su uso en matplotlib
        words = top_20_words_df.select("word").rdd.flatMap(lambda x: x).collect()
        counts = top_20_words_df.select("count").rdd.flatMap(lambda x: x).collect()
        
        # Graficar las 20 palabras más frecuentes
        plt.figure(figsize=(10, 6))
        plt.barh(words, counts, color='skyblue')
        plt.xlabel("Frequency")
        plt.ylabel("Words")
        plt.title("Top 20 Most Frequent Words")
        plt.gca().invert_yaxis()  # Para que las palabras más frecuentes estén en la parte superior
        plt.show()
