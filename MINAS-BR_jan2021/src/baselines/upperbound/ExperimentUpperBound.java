package baselines.upperbound;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.yahoo.labs.samoa.instances.Prediction;
import dataSource.DataSetUtils;
import evaluate.Evaluator;
import evaluate.EvaluatorBR;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import meka.classifiers.multilabel.incremental.BRUpdateable;
import meka.classifiers.multilabel.incremental.CCUpdateable;
import moa.classifiers.MultiLabelLearner;
import moa.classifiers.multilabel.MEKAClassifier;
import moa.classifiers.multilabel.MultilabelHoeffdingTree;
import moa.classifiers.multilabel.meta.OzaBagAdwinML;
import moa.classifiers.multilabel.trees.ISOUPTree;
import moa.core.InstanceExample;
import moa.streams.MultiTargetArffFileStream;
import utils.FilesOutput;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.trees.HoeffdingTree;

/**
 *
 * @author Joel
 */
public class ExperimentUpperBound {
    
    public static void execute(String fileTrain,String fileTest, String outputDirectory, int numWindows ) throws Exception {
            runExperiment(fileTrain, fileTest, numWindows, outputDirectory);
    }
    public static void execute(int numWindows) throws Exception{
        
            String dataSetName = "com_NP/SynT";
            String fileTrain = "D:\\Google Drive\\datasets\\datasets_sinteticos\\MultiLabelGenerator\\RTG\\SynT_drift_novelty\\SynT_drift_novelty_train.arff";
            String fileTest = "D:\\Google Drive\\datasets\\datasets_sinteticos\\MultiLabelGenerator\\RTG\\SynT_drift_novelty\\SynT_drift_novelty_test.arff";
            String outputDirectory = "experiments/upperBoundMethod/"+dataSetName+"/";
            runExperiment(fileTrain, fileTest, numWindows, outputDirectory);
            
//            String dataSetName = "sem_NP/4CRE-V2";
//            int numLabels = 2;
//            String fileTrain = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\4CRE-V2_train.arff";
//            String fileTest = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\4CRE-V2_test.arff";
//            String outputDirectory = "experiments/upperBoundMethod/"+dataSetName+"/";
//            runExperiment(fileTrain, fileTest, numWindows, numLabels, outputDirectory);
            
//            dataSetName = "sem_NP/MOA-3C-5C-2D";
//            numLabels = 3;
//            fileTrain = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\MOA-3C-5C-2D_train.arff";
//            fileTest = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\\\MOA-3C-5C-2D_test.arff";
//            outputDirectory = "experiments/upperBoundMethod/"+dataSetName+"/";
//            runExperiment(fileTrain, fileTest, numWindows, numLabels, outputDirectory);
//            
//            dataSetName = "sem_NP/MOA-5C-7C-2D";
//            numLabels = 5;
//            fileTrain = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\\\MOA-5C-7C-2D_train.arff";
//            fileTest = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\MOA-5C-7C-2D_test.arff";
//            outputDirectory = "experiments/upperBoundMethod/"+dataSetName+"/";
//            runExperiment(fileTrain, fileTest, numWindows,numLabels, outputDirectory);
            
//            String dataSetName = "sem_NP/MOA-5C-7C-3D";
//            int numLabels = 5;
//            String fileTrain = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\\\MOA-5C-7C-3D_train.arff";
//            String fileTest = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\MOA-5C-7C-3D_test.arff";
//            String outputDirectory = "experiments/upperBoundMethod/"+dataSetName+"/";
//            runExperiment(fileTrain, fileTest, numWindows, numLabels, outputDirectory);
            
//            String dataSetName = "sem_NP/SynHyperPlane";
//            int numLabels = 5;
//            String fileTrain = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\SynHyperPlane_drift_novelty_train.arff";
//            String fileTest = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\SynHyperPlane_drift_novelty_test.arff";
//            String outputDirectory = "experiments/upperBoundMethod/"+dataSetName+"/";
//            runExperiment(fileTrain, fileTest, numWindows, numLabels, outputDirectory);
            
//            String dataSetName = "sem_NP/SynWaveForm";
//            int numLabels = 7;
//            String fileTrain = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\SynWaveForm_drift_novelty_train.arff";
//            String fileTest = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\SynWaveForm_drift_novelty_test.arff";
//            String outputDirectory = "experiments/upperBoundMethod/"+dataSetName+"/";
//            runExperiment(fileTrain, fileTest, numWindows, numLabels, outputDirectory);
//            
//            dataSetName = "sem_NP/MOA-5C-7C-3D";
//            fileTrain = "D:\\Google Drive\\datasets\\datasets_Sem_NP_2\\Datasets_Sem_NP\\MOA-5C-7C-3D-train.arff";
//            fileTest = "D:\\Google Drive\\datasets\\datasets_Sem_NP_2\\Datasets_Sem_NP\\MOA-5C-7C-3D-test.arff";
//            outputDirectory = "experiments/batchMethods/"+dataSetName+"/";
//            ExperimentBatch.runExperiment(fileTrain, fileTest, numWindows, outputDirectory);
//            
//            dataSetName = "sem_NP/nus-wide";
//            fileTrain = "D:\\Google Drive\\datasets\\datasets_Sem_NP_2\\Datasets_Sem_NP\\nus-wide_modified-train.arff";
//            fileTest = "D:\\Google Drive\\datasets\\datasets_Sem_NP_2\\Datasets_Sem_NP\\nus-wide_modified-test.arff";
//            outputDirectory = "experiments/batchMethods/"+dataSetName+"/";
//            ExperimentBatch.runExperiment(fileTrain, fileTest, numWindows, outputDirectory);
    }
    
    public static void runExperiment(String fileTrain, String fileTest, int numWindows, String outputDirectory) throws Exception{
        int numLabels = DataSetUtils.getNumLabels(fileTrain);
        ArrayList<InstanceExample> trainSet = new ArrayList<>();
        ArrayList<InstanceExample> testSet = new ArrayList<>();
        InstancesHeader header = getHeaderLoadDataSet(fileTrain,numLabels,trainSet);
        getHeaderLoadDataSet(fileTest,numLabels,testSet);
        ArrayList<Evaluator> evaluatorList = new ArrayList<Evaluator>();
        
        String method = "EaBR";
//        evaluatorList.add(executeMethod(method, header, trainSet, testSet, numWindows, numLabels, outputDirectory));
//        method = "EaCC";
//        evaluatorList.add(executeMethod(method, header, trainSet, testSet, numWindows, numLabels,outputDirectory));
//        method = "CC";
//        evaluatorList.add(executeMethod(method, header, trainSet, testSet, numWindows, numLabels,outputDirectory));
//        method = "BR";
//        evaluatorList.add(executeMethod(method, header, trainSet, testSet, numWindows, numLabels,outputDirectory));
//        method = "EaPS";
//        evaluatorList.add(executeMethod(method, header, trainSet, testSet, numWindows, numLabels,outputDirectory));
////        method = "PS";
//        evaluatorList.add(executeMethod(method, header, trainSet, testSet, numWindows, numLabels,outputDirectory));
//        method = "MLHT";
//        evaluatorList.add(executeMethod(method, header, trainSet, testSet, numWindows, numLabels,outputDirectory));
        method = "EaMLHT";
        evaluatorList.add(executeMethod(method, header, trainSet, testSet, numWindows, numLabels,outputDirectory));
//        method = "ISOUPTree";
//        evaluatorList.add(executeMethod(method, header, trainSet, testSet, numWindows, numLabels,outputDirectory));
        method = "EaISOUPTree";
        evaluatorList.add(executeMethod(method, header, trainSet, testSet, numWindows, numLabels,outputDirectory));
        
        EvaluatorBR.writesAvgResults(evaluatorList, outputDirectory);
    }
    
    public static InstancesHeader getHeaderLoadDataSet (String file, int numLabels, ArrayList<InstanceExample> instances){
        MultiTargetArffFileStream dataset = new MultiTargetArffFileStream(file, String.valueOf(numLabels));
        dataset.prepareForUse();
        while(dataset.hasMoreInstances()){
            instances.add(dataset.nextInstance());
        }
        return dataset.getHeader();
    }
    
    /**
     * Preperares a method to run an experiment
     * @param method
     * @param header
     * @param numLabels
     * @param knownClasses
     * @return 
     */
    public static MultiLabelLearner getLearner(String method, InstancesHeader header, int numLabels, Set<String> knownClasses){
        if(method.equals("MLHT")){
            MultilabelHoeffdingTree learner = new MultilabelHoeffdingTree();
            MEKAClassifier PS = new MEKAClassifier();
            PSUpdateableModified base = new PSUpdateableModified();
            PS.baseLearnerOption.setCurrentObject(base);
            PS.setModelContext(header);
            PS.prepareForUse();
            PS.resetLearningImpl();
            learner.learnerOption.setCurrentObject(PS);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;
            
        }else if(method.equals("BR")){
            MEKAClassifier learner = new MEKAClassifier();
            learner.setModelContext(header);
            learner.prepareForUse();
            return learner;
            
        }else if(method.equals("PS")){
            MEKAClassifier learner = new MEKAClassifier();
            PSUpdateableModified baseLearner = new PSUpdateableModified();
            learner.baseLearnerOption.setCurrentObject(baseLearner);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;
            
        }else if(method.equals("CC")){
            MEKAClassifier learner = new MEKAClassifier();
            CCUpdateable baseLearner = new CCUpdateable();
            learner.baseLearnerOption.setCurrentObject(baseLearner);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;
            
        }else if(method.equals("EaBR")){
            OzaBagAdwinML learner = new OzaBagAdwinML();
            MEKAClassifier baseMeka = new MEKAClassifier();
            BRUpdateable br = new BRUpdateable();
            baseMeka.baseLearnerOption.setCurrentObject(br);
            baseMeka.setModelContext(header);
            baseMeka.prepareForUse();
            baseMeka.resetLearningImpl();
            learner.baseLearnerOption.setCurrentObject(baseMeka);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;
            
        }else if(method.equals("EaMLHT")){
            OzaBagAdwinML learner = new OzaBagAdwinML();
            MultilabelHoeffdingTree MLHT = new MultilabelHoeffdingTree();
            MEKAClassifier PS = new MEKAClassifier();
            PSUpdateableModified psUpdateable = new PSUpdateableModified();
            PS.baseLearnerOption.setCurrentObject(psUpdateable);
            PS.setModelContext(header);
            PS.prepareForUse();
            PS.resetLearningImpl();
            MLHT.learnerOption.setCurrentObject(PS);
            MLHT.setModelContext(header);
            MLHT.prepareForUse();
            MLHT.resetLearningImpl();
            learner.baseLearnerOption.setCurrentObject(MLHT);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;
            
        }else if(method.equals("EaCC")){
            OzaBagAdwinML learner = new OzaBagAdwinML();
            MEKAClassifier baseMeka = new MEKAClassifier();
            CCUpdateable baseLearner = new CCUpdateable();
            baseMeka.baseLearnerOption.setCurrentObject(baseLearner);
            baseMeka.setModelContext(header);
            baseMeka.prepareForUse();
            baseMeka.resetLearningImpl();
            learner.baseLearnerOption.setCurrentObject(baseMeka);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;
            
        }else if(method.equals("EaPS")){
            OzaBagAdwinML learner = new OzaBagAdwinML();
            MEKAClassifier baseMeka = new MEKAClassifier();
            PSUpdateableModified baseLearner = new PSUpdateableModified();
            baseMeka.baseLearnerOption.setCurrentObject(baseLearner);
            baseMeka.setModelContext(header);
            baseMeka.prepareForUse();
            baseMeka.resetLearningImpl();
            learner.baseLearnerOption.setCurrentObject(baseMeka);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;
        }else if(method.equals("ISOUPTree")){
            ISOUPTree learner = new ISOUPTree();
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;
        }else if(method.equals("EaISOUPTree")){
            OzaBagAdwinML learner = new OzaBagAdwinML();
            ISOUPTree base = new ISOUPTree();
            base.setModelContext(header);
            base.prepareForUse();
            base.resetLearningImpl();
            learner.baseLearnerOption.setCurrentObject(base);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;
        }
        
        return null;
      
    }
    
    /**
     * Run an specific method and return results into a Evaluator object
     * @param method
     * @param header
     * @param trainSet
     * @param testSet
     * @param numWindows
     * @param numLabels
     * @return
     * @throws IOException 
     */
    public static Evaluator executeMethod(String method, InstancesHeader header, ArrayList<InstanceExample> trainSet, ArrayList<InstanceExample> testSet, int numWindows, int numLabels, String outputDirectory) throws IOException{
        String output = outputDirectory+"/"+method+"/";
        FilesOutput.createDirectory(output);
        
        FileWriter filePredictions = new FileWriter(new File(output+"prediction.txt"), false);
        filePredictions.write("Trainset size: " + trainSet.size() + "\n");
        System.out.println("Trainset size: " + trainSet.size() + "\n");
        filePredictions.write("Testset size: " + testSet.size() + "\n");
        System.out.println("Testset size: " + testSet.size() + "\n");
        
        Set<String> knownClasses = DataSetUtils.getClassesConhecidas((List)trainSet, numLabels);
//        filePredictions.write("Known Classes: "+knownClasses + "\n");
        System.out.println("Known Classes: "+knownClasses + "\n");;;
        float evaluationWindowsSizeAux = (float)testSet.size()/(float)numWindows;
        int evaluationWindowsSize = (int)Math.ceil(evaluationWindowsSizeAux);
        filePredictions.write("Evaluation windows size: " + evaluationWindowsSize + "\n");
        System.out.println("Evaluation windows size: " + evaluationWindowsSize + "\n");
        ArrayList<double[]> windowsCardinalities = DataSetUtils.getWindowsCardinalities(testSet, evaluationWindowsSize,numWindows, numLabels);
        Evaluator av = new Evaluator(numLabels, knownClasses, method);
        MultiLabelLearner learner = getLearner(method, header, numLabels, knownClasses);
        
        ArrayList<Prediction> predictionlist = new ArrayList<>();
        ArrayList<Set<String>> trueLabelsList = new ArrayList<>();
        
        System.out.println("###Offline Phase####");
        for(int i = 0; i < trainSet.size();i++){
            System.out.println("Train: " + i);
//            if(i == 3549){
//                System.out.println("Pula");
//                continue;
//            }
            System.out.println("Size:" + trainSet.get(i).instance.toDoubleArray().length);
            learner.trainOnInstance(trainSet.get(i));
        }
        
        System.out.println("###Online Phase####");
        int window = 1;
        int i = 0;
        while (i < testSet.size()) {
            trueLabelsList.add(DataSetUtils.getLabelSet(testSet.get(i).getData()));
            Prediction predictions = learner.getPredictionForInstance(testSet.get(i));
            predictionlist.add(predictions);
            learner.trainOnInstance(testSet.get(i));
            filePredictions.write("True Labels: " + trueLabelsList.get(trueLabelsList.size()-1) + "\t Predicted: " + predictions.toString() + "\n");

            if (i > 0 && i % evaluationWindowsSize == 0){
                System.out.println("####Window - " + window + "####");
                av.updateMeasuresThresholding(predictionlist,  trueLabelsList, windowsCardinalities.get(window-1));
                window++;
                predictionlist.clear();
                trueLabelsList.clear();
            }
            i++;
        }
        if(!predictionlist.isEmpty()){
            System.out.println("####Window - " + window + "####");
            av.updateMeasuresThresholding(predictionlist,  trueLabelsList, windowsCardinalities.get(window-1));
            window++;
            predictionlist.clear();
            trueLabelsList.clear();
        }
        filePredictions.close();
        av.writeMeasuresOverTime(output);
        return av;
    }
    
    
}
