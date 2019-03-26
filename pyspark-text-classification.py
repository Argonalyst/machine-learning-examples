# Simple script o classify texts using pyspark

from __future__ import print_function
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("CrossValidatorExample")\
        .getOrCreate()

    # Prepare training documents, which are labeled.
    # religion vs. games
    training = spark.createDataFrame([
        (0, "jesus cristo é filho de deus", 1.0),
        (1, "eu gosto de jogos de tabuleiro", 0.0),
        (2, "santo pai de todos os anjos", 1.0),
        (3, "jogadores de computador participam de campeonato", 0.0),
        (4, "deus na terra", 1.0),
        (5, "os video game nintendo switch é legal", 0.0),
        (6, "jesus cristo é o senhor", 1.0),
        (7, "god of war ganhou prêmio de melhor jogo de 2018", 0.0),
        (8, "pai, filho e espírito santo", 1.0),
        (9, "jogar muito estimula o cêrebro", 0.0),
        (10, "ingrejas e padres são coisas religiosas", 1.0),
        (11, "jogos são legais", 0.0)
    ], ["id", "text", "label"])

    # Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and lr.
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=10)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

    paramGrid = ParamGridBuilder() \
        .addGrid(hashingTF.numFeatures, [10, 100, 1000]) \
        .addGrid(lr.regParam, [0.1, 0.01]) \
        .build()

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=BinaryClassificationEvaluator(),
                              numFolds=3)

    cvModel = crossval.fit(training)

    test = spark.createDataFrame([
        (4, "eu gosto de jogar"),
        (5, "faz tempo que não vou a igreja"),
        (6, "jesus cristo"),
        (7, "muitos jogos legais lançados recentemente")
    ], ["id", "text"])

    prediction = cvModel.transform(test)
    selected = prediction.select("id", "text", "probability", "prediction")
    for row in selected.collect():
        print(row)

    spark.stop()