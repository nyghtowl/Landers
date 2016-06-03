package org.deeplearning4j.java.example;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.canova.api.io.filters.BalancedPathFilter;
import org.canova.api.io.labels.ParentPathLabelGenerator;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.image.loader.BaseImageLoader;
import org.canova.image.recordreader.ImageRecordReader;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.List;
import java.util.Random;

public class JavaImgClassification {
    protected static final Logger log = LoggerFactory.getLogger(JavaImgClassification.class);
    protected static long seed = 123;
    protected static int height = 50;
    protected static int width = 50;
    protected static int channels = 3;
    protected static int numExamples = 80;
    protected static int outputNum = 4;
    protected static int batchSize = 20;
    protected static int listenerFreq = 1;
    protected static int iterations = 1;
    protected static int epochs = 1;

    public static void main(String[] args) throws Exception {
        File mainPath = new File(System.getProperty("user.home"), "data/animals");
        RecordReader recordReader = new ImageRecordReader(width, height, channels, new ParentPathLabelGenerator());
        FileSplit fileSplit = new FileSplit(mainPath, BaseImageLoader.ALLOWED_FORMATS, new Random(123));
        BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(123), BaseImageLoader.ALLOWED_FORMATS, new ParentPathLabelGenerator(), numExamples, outputNum, 0, batchSize);
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, 80, 20);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        recordReader.initialize(trainData);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

        // Tiny Config
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
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(height, width, channels).build();

        MultiLayerNetwork network = new MultiLayerNetwork(confTiny);
        network.init();
        network.setListeners(new ScoreIterationListener(listenerFreq));

        MultipleEpochsIterator multDataIter = new MultipleEpochsIterator(epochs, dataIter);
        network.fit(multDataIter);

        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, 20, 1, outputNum);
        DataSet testDataSet = dataIter.next();
        Evaluation eval = new Evaluation(recordReader.getLabels());
        eval.eval(testDataSet.getLabels(), network.output(testDataSet.getFeatureMatrix(), Layer.TrainingMode.TEST));
        log.info(eval.stats(true));

        // Check prediction
        String expectedResult = testDataSet.getLabelName(0);
        List<String> predict = network.predict(testDataSet);
        String result = predict.get(0);

        String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
        String confPath = FilenameUtils.concat(basePath, "TinyModel-conf.json");
        String paramPath = FilenameUtils.concat(basePath, "TinyModel-params.bin");

        DataOutputStream dos = new DataOutputStream(new FileOutputStream(paramPath));
        Nd4j.write(network.params(), dos);
        dos.flush();
        dos.close();
        // save model configuration
        FileUtils.write(new File(confPath), network.conf().toJson());
    }

}
