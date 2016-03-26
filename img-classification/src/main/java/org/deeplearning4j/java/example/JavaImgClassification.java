package org.deeplearning4j.java.example;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.LimitFileSplit;
import org.canova.image.loader.BaseImageLoader;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

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
    protected static boolean appendLabels = true;
    protected static int iterations = 2;
    protected static int epochs = 2;
    protected static int splitTrainNum = 10;

    public static void main(String[] args) throws Exception {
        File mainPath = new File(System.getProperty("user.home"), "animals");
        List<String> labels = Arrays.asList(new String[]{"bear", "deer", "duck", "turtle"});
        RecordReader recordReader = new ImageRecordReader(width, height, channels, appendLabels);

        recordReader.initialize(new LimitFileSplit(mainPath, BaseImageLoader.ALLOWED_FORMATS, numExamples, outputNum, null, new Random(123)));

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, -1, outputNum);

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
                .cnnInputSize(height, width, channels).build();

        // AlexNet Config
        int nonZeroBias = 1;
        double dropOut = 0.5;
        SubsamplingLayer.PoolingType poolingType = SubsamplingLayer.PoolingType.MAX;

        MultiLayerConfiguration confAlexNet = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .updater(Updater.NESTEROVS)
                .iterations(iterations)
                // normalize to prevent vanishing or exploding gradients
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-3)
                .learningRateScoreBasedDecayRate(1e-1)
                .regularization(true)
                .l2(5 * 1e-4)
                .momentum(0.9)
                .miniBatch(false)
                .list(13)
                //conv1
                .layer(0, new ConvolutionLayer.Builder(new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3})
                        .name("cnn1")
                        .nIn(channels)
                        .nOut(96)
                        .build())
                .layer(1, new LocalResponseNormalization.Builder()
                        .name("lrn1")
                        .build())
                .layer(2, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .name("pool1")
                        .build())
                //conv2
                .layer(3, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{2, 2})
                        .name("cnn2")
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(4, new LocalResponseNormalization.Builder()
                        .name("lrn2")
                        .k(2).n(5).alpha(1e-4).beta(0.75)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .name("pool2")
                        .build())
                //conv3
                .layer(6, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn3")
                        .nOut(384)
                        .build())
                //conv4
                .layer(7, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn4")
                        .nOut(384)
                        .biasInit(nonZeroBias)
                        .build())
                //conv5
                .layer(8, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn5")
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(9, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .name("pool3")
                        .build())
                .layer(10, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(4096)
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(11, new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(4096)
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true)
                .pretrain(false)
                .cnnInputSize(height,width,channels)
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(confTiny);
        network.init();
        network.setListeners(new ScoreIterationListener(listenerFreq));

        SplitTestAndTrain trainTest;
        DataSet trainInput;
        List<INDArray> testInput = new ArrayList<INDArray>();
        List<INDArray> testLabels = new ArrayList<INDArray>();
        DataSet dsNext;

        while (dataIter.hasNext()) {
            dsNext = dataIter.next();
            dsNext.scale();
            trainTest = dsNext.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            trainInput = trainTest.getTrain(); // get feature matrix and labels for training
            testInput.add(trainTest.getTest().getFeatureMatrix());
            testLabels.add(trainTest.getTest().getLabels());
            network.fit(trainInput);
        }

        // more than 1 epoch for just training
        for(int i = 1; i < epochs; i++) {
            dataIter.reset();
            while (dataIter.hasNext()) {
                dsNext = dataIter.next();
                trainTest = dsNext.splitTestAndTrain(splitTrainNum, new Random(seed));
                trainInput = trainTest.getTrain();
                network.fit(trainInput);
            }
        }

        Evaluation eval = new Evaluation(labels);
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = network.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }
        INDArray output = network.output(testInput.get(0));
        eval.eval(testLabels.get(0), output);
        log.info(eval.stats());

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
