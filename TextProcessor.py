from pyspark.sql.functions import split, col

class TextProcessor:
    def __init__(self, spark):
        self.spark = spark

    def clean_text(self, text):
        # Limpieza básica del texto
        return text

    def tokenize_text(self, cleaned_text):
        # Crear un DataFrame con el texto limpio
        df = self.spark.createDataFrame([(cleaned_text,)], ["text"])
        # Dividir el texto en palabras (por ejemplo, usando split por espacios)
        words_df = df.withColumn("words", split(col("text"), " "))
        return words_df
