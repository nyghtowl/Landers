package org.deeplearning4j.java.example;

import org.apache.commons.io.FilenameUtils;
import org.canova.api.io.filters.BalancedPathFilter;
import org.canova.api.io.labels.ParentPathLabelGenerator;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.image.loader.BaseImageLoader;
import org.canova.image.recordreader.ImageRecordReader;
import org.canova.image.transform.FlipImageTransform;
import org.canova.image.transform.ImageTransform;
import org.canova.image.transform.MultiImageTransform;
import org.canova.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.NetSaverLoaderUtils;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.Random;

public class JavaImgClassification {
    protected static final Logger log = LoggerFactory.getLogger(JavaImgClassification.class);

    protected static int seed = 42;
    protected static int height = 50;
    protected static int width = 50;
    protected static int channels = 3;
    protected static int numExamples = 80;
    protected static int numLabels = 4;
    protected static int batchSize = 20;
    protected static int listenerFreq = 1;
    protected static int iterations = 1;
    protected static int epochs = 1;
    protected static double splitTrainTest = 0.8;
    protected static Random rng = new Random(seed);

    public static void main(String[] args) throws Exception {

        log.info("Load data....");
        /**
         * Data Setup -> organize and limit data file paths:
         *  - mainPath = path to image files
         *  - fileSplit = define basic dataset split with limits on format
         *  - pathFilter = define additional file load filter to limit size and balance batch content
         **/
        File mainPath = new File(System.getProperty("user.dir"), "src/main/resources/");
        FileSplit fileSplit = new FileSplit(mainPath, BaseImageLoader.ALLOWED_FORMATS, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, BaseImageLoader.ALLOWED_FORMATS, new ParentPathLabelGenerator(), numExamples, numLabels, 0, batchSize);

        /**
         * Data Setup -> train test split
         *  - inputSplit = define train and test split
         **/
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, numExamples*(1+splitTrainTest),  numExamples*(1-splitTrainTest));
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        /**
         * Data Setup -> define how to load data into net:
         *  - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
         *  - dataIter = a generator that only loads one batch at a time into memory to save memory
         *  - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all epochs
         **/
        RecordReader recordReader = new ImageRecordReader(width, height, channels, new ParentPathLabelGenerator());
        recordReader.initialize(trainData);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        MultipleEpochsIterator trainIter = new MultipleEpochsIterator(epochs, dataIter);

        log.info("Build model....");
        // Tiny model configuration
        MultiLayerConfiguration confTiny = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("relu")
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .learningRate(0.01)
                .momentum(0.9)
                .regularization(true)
                .l2(0.04)
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
                .cnnInputSize(height, width, channels).build();

        MultiLayerNetwork network = new MultiLayerNetwork(confTiny);
        network.init();
        network.setListeners(new ScoreIterationListener(listenerFreq));

        log.info("Train model....");
        network.fit(trainIter);

        log.info("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, 20, 1, numLabels);
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats(true));

        // Example on how to get predict results with trained model
        dataIter.reset();
        DataSet testDataSet = dataIter.next();
        String expectedResult = testDataSet.getLabelName(0);
        List<String> predict = network.predict(testDataSet);
        String modelResult = predict.get(0);
        System.out.print("For a single example that is labeled " + expectedResult+ " the model predicted " + modelResult + "\n");

        log.info("Save model....");
        String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
        NetSaverLoaderUtils.saveNetworkAndParameters(network, basePath);
        NetSaverLoaderUtils.saveUpdators(network, basePath);

        log.info("****************Example finished********************");

    }
}
