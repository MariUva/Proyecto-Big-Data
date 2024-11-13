from pyspark.sql.functions import explode, col

class WordFrequencyAnalyzer:
    def __init__(self, spark):
        self.spark = spark

    def analyze_frequency(self, words_df):
        # Separar cada palabra en una fila
        exploded_df = words_df.select(explode(col("words")).alias("word"))
        # Contar la frecuencia de cada palabra
        word_counts_df = exploded_df.groupBy("word").count().orderBy(col("count").desc())
        return word_counts_df.limit(20)  # Limitar a las 20 palabras más frecuentes
