package org.deeplearning4j.scala.example

import java.io.{File, IOException}
import java.util
import java.util.Random

import org.apache.commons.io.{FilenameUtils}
import org.canova.api.io.filters.BalancedPathFilter
import org.canova.api.io.labels.ParentPathLabelGenerator
import org.canova.api.records.reader.RecordReader
import org.canova.api.split.{FileSplit, InputSplit}
import org.canova.image.loader.BaseImageLoader
import org.canova.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.{DataSetIterator, MultipleEpochsIterator}
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, LocalResponseNormalization, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.NetSaverLoaderUtils
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions


object ScalaImgClasssification {

  val seed = 42
  val height = 50
  val width = 50
  val channels = 3
  val numExamples = 80
  val numLabels = 4
  val batchSize = 20
  val listenerFreq = 1
  val iterations = 1
  val epochs = 5
  val splitTrainTest = 0.8
  val rng = new Random(seed);


  def main(args: Array[String]) {

    val basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/")
    val mainPath: File = new File(basePath, "animals")

    // Define how to filter and load data into batches
    val recordReader: RecordReader = new ImageRecordReader(width, height, channels, new ParentPathLabelGenerator())
    val fileSplit: FileSplit = new FileSplit(mainPath, BaseImageLoader.ALLOWED_FORMATS,rng)
    val pathFilter: BalancedPathFilter = new BalancedPathFilter(rng, BaseImageLoader.ALLOWED_FORMATS, new ParentPathLabelGenerator, numExamples, numLabels, 0, batchSize)

    // Define train and test split
    val inputSplit: Array[InputSplit] = fileSplit.sample(pathFilter, numExamples * (1 + splitTrainTest), numExamples * (1 - splitTrainTest))
    val trainData: InputSplit = inputSplit(0)
    val testData: InputSplit = inputSplit(1)

    // Define how to load data into network
    try {
      recordReader.initialize(trainData)
    } catch {
      case ioe: IOException => ioe.printStackTrace()
      case e: InterruptedException => e.printStackTrace()
    }
    var dataIter: DataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)
    val multDataIter: MultipleEpochsIterator = new MultipleEpochsIterator(epochs, dataIter)

    // Build model based on tiny model configuration paper
    val confTiny: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .activation("relu")
      .weightInit(WeightInit.XAVIER)
      .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .updater(Updater.NESTEROVS)
      .learningRate(0.01)
      .momentum(0.9)
      .regularization(true)
      .l2(0.04)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .useDropConnect(true)
      .list()
      .layer(0, new ConvolutionLayer.Builder(5, 5)
        .name("cnn1")
        .nIn(channels)
        .stride(1, 1)
        .padding(2, 2)
        .nOut(32)
        .build())
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(3, 3)
        .name("pool1")
        .build())
      .layer(2, new LocalResponseNormalization.Builder(3, 5e-05, 0.75).build())
      .layer(3, new ConvolutionLayer.Builder(5, 5)
        .name("cnn2")
        .stride(1, 1)
        .padding(2, 2)
        .nOut(32)
        .build())
      .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(3, 3)
        .name("pool2")
        .build())
      .layer(5, new LocalResponseNormalization.Builder(3, 5e-05, 0.75).build())
      .layer(6, new ConvolutionLayer.Builder(5, 5)
        .name("cnn3")
        .stride(1, 1)
        .padding(2, 2)
        .nOut(64)
        .build())
      .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(3, 3)
        .name("pool3")
        .build())
      .layer(8, new DenseLayer.Builder()
        .name("ffn1")
        .nOut(250)
        .dropOut(0.5)
        .build())
      .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(numLabels)
        .activation("softmax")
        .build())
      .backprop(true).pretrain(false)
      .cnnInputSize(height, width, channels).build()

    val network: MultiLayerNetwork = new MultiLayerNetwork(confTiny)
    network.init()
    network.setListeners(new ScoreIterationListener(listenerFreq))

    // Train
    network.fit(multDataIter)

    // Evaluate
    recordReader.initialize(testData)
    dataIter = new RecordReaderDataSetIterator(recordReader, 20, 1, numLabels)
    val eval = network.evaluate(dataIter)
    print(eval.stats(true))

    // Predict
    dataIter.reset()
    val testDataSet: DataSet = dataIter.next
    val expectedResult: String = testDataSet.getLabelName(0)
    val predict: util.List[String] = network.predict(testDataSet)
    val modelResult: String = predict.get(0)
    System.out.print("For a single example that is labeled " + expectedResult + " the model predicted " + modelResult + "\n")

    // Save model and parameters
    NetSaverLoaderUtils.saveNetworkAndParameters(network, basePath)
    NetSaverLoaderUtils.saveUpdators(network, basePath)

  }
}

