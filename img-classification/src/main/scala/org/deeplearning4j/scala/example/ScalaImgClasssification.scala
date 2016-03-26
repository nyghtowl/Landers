package org.deeplearning4j.scala.example

import java.io.{DataOutputStream, File, FileOutputStream, IOException}
import java.util.Random

import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.canova.api.records.reader.RecordReader
import org.canova.api.split.LimitFileSplit
import org.canova.image.loader.BaseImageLoader
import org.canova.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{GradientNormalization, MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, LocalResponseNormalization, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.{DataSet, SplitTestAndTrain}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.collection.mutable.ListBuffer
import collection.JavaConversions._

object ScalaImgClasssification {

  val seed = 123
  val height = 50
  val width = 50
  val channels = 3
  val numExamples = 80
  val outputNum = 4
  val batchSize = 20
  val listenerFreq = 5
  val appendLabels = true
  val iterations = 2
  val epochs = 2
  val splitTrainNum = 10


  def main(args: Array[String]) {
    val basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/")
    val mainPath: File = new File(basePath, "animals")
    val labels: List[String] = List("bear", "deer", "duck", "turtle")

    val recordReader: RecordReader = new ImageRecordReader(width, height, channels, appendLabels)
    try {
      recordReader.initialize(
        new LimitFileSplit(mainPath, BaseImageLoader.ALLOWED_FORMATS, numExamples, outputNum, null, new Random(123)))
    } catch {
      case ioe: IOException => ioe.printStackTrace()
      case e: InterruptedException => e.printStackTrace()
    }
    val dataIter: DataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, -1, outputNum)


    val confTiny: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .activation("relu")
      .weightInit(WeightInit.XAVIER)
      .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .updater(Updater.SGD)
      .learningRate(0.01)
      .momentum(0.9)
      .regularization(true)
      .l2(0.04)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .list(10)
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
        .nOut(outputNum)
        .activation("softmax")
        .build())
      .backprop(true).pretrain(false)
      .cnnInputSize(height, width, channels).build()


    val network: MultiLayerNetwork = new MultiLayerNetwork(confTiny)
    network.init()

    network.setListeners(new ScoreIterationListener(listenerFreq))

    val testInput = new ListBuffer[INDArray]()
    val testLabels = new ListBuffer[INDArray]()

    while (dataIter.hasNext()) {
      val dsNext: DataSet = dataIter.next()
      dsNext.scale()
      val trainTest: SplitTestAndTrain = dsNext.splitTestAndTrain(splitTrainNum, new Random(seed))
      val trainInput: DataSet = trainTest.getTrain() // get feature matrix and labels for training
      testInput += trainTest.getTest().getFeatureMatrix()
      testLabels += trainTest.getTest().getLabels()
      network.fit(trainInput)
    }

    // Assumes 1 epoch completed already
    for (i <- 1 until epochs) {
      dataIter.reset()
      while (dataIter.hasNext()) {
        val dsNext: DataSet = dataIter.next()
        val trainTest: SplitTestAndTrain = dsNext.splitTestAndTrain(splitTrainNum, new Random(seed))
        val trainInput: DataSet = trainTest.getTrain()
        network.fit(trainInput)
      }
    }

    val eval = new Evaluation(labels)
    for(i <- 0 until testInput.length) {
      val output: INDArray = network.output(testInput.get(i))
      eval.eval(testLabels.get(i), output)
    }
    print(eval.stats())

    val confPath = FilenameUtils.concat(basePath, "TinyModel-conf.json")
    val paramPath = FilenameUtils.concat(basePath, "TinyModel-params.bin")

    // save parameters
    try {
      val dos: DataOutputStream = new DataOutputStream(new FileOutputStream(paramPath))
      Nd4j.write(network.params(), dos)
      dos.flush()
      dos.close()
      // save model configuration
      FileUtils.write(new File(confPath), network.conf().toJson())
    } catch {
      case ioe: IOException => ioe.printStackTrace()
    }
  }
}

