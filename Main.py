from pyspark.sql import SparkSession
from PDFReader import PDFReader
from TextProcessor import TextProcessor
from WordFrequencyAnalyzer import WordFrequencyAnalyzer
from WordFrequencyPlotter import WordFrequencyPlotter
from pyspark.sql.functions import explode, col

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class Main:
    def __init__(self, pdf_path):
        # Establecer la ruta del archivo PDF
        self.pdf_path = pdf_path
        # Crear sesión de Spark
        self.spark = SparkSession.builder.appName("WordFrequencyPDF").getOrCreate()
        # Inicializar las clases para el procesamiento
        self.pdf_reader = PDFReader(pdf_path)
        self.text_processor = TextProcessor(self.spark)
        self.frequency_analyzer = WordFrequencyAnalyzer(self.spark)
        self.plotter = WordFrequencyPlotter()

    def run(self):
        # Paso 1: Leer el PDF
        text = self.pdf_reader.read_pdf()
        
        # Paso 2: Limpiar y tokenizar el texto
        cleaned_text = self.text_processor.clean_text(text)
        words_df = self.text_processor.tokenize_text(cleaned_text)
        
        # Paso 3: Analizar la frecuencia de las palabras
        top_20_words = self.frequency_analyzer.analyze_frequency(words_df)
        
        # Paso 4: Graficar las 20 palabras más frecuentes
        self.plotter.plot_top_20_words(top_20_words)
        
        # Paso 5: Crear una nube de palabras
        self.generate_wordcloud(cleaned_text)
        
        # Paso 6: Aplicar TfidfVectorizer y mostrar los resultados
        self.apply_tfidf_vectorizer(cleaned_text)

    def generate_wordcloud(self, text):
        # Crear y mostrar la nube de palabras
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
    
    def apply_tfidf_vectorizer(self, text):
        # Usar TfidfVectorizer para el texto procesado
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text])
        
        # Mostrar algunas palabras clave con mayor peso TF-IDF
        feature_names = vectorizer.get_feature_names_out()
        dense = tfidf_matrix.todense().tolist()[0]
        tfidf_dict = dict(zip(feature_names, dense))
        
        # Ordenar por peso de TF-IDF y mostrar las palabras con mayor peso
        sorted_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        print("Palabras clave con mayor peso TF-IDF:")
        for word, score in sorted_tfidf:
            print(f"{word}: {score}")

if __name__ == "__main__":
    # Ruta del archivo PDF (reemplaza con la ruta de tu archivo PDF)
    pdf_path = "C:/Users/maria/Downloads/cuento.pdf"
    main_program = Main(pdf_path)
    main_program.run()
