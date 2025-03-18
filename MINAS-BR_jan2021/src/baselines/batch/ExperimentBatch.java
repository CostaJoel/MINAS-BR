package baselines.batch;

/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
import dataSource.DataSetUtils;
import meka.core.MLEvalUtils;
import meka.core.MLUtils;
import meka.core.Result;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.util.Set;
import meka.classifiers.multilabel.MultiLabelClassifier;
import meka.core.ThresholdUtils;
import evaluate.Evaluator;
import evaluate.EvaluatorBR;
import java.util.ArrayList;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.DBPNN;
import meka.classifiers.multilabel.FW;
import meka.classifiers.multilabel.MULAN;
import meka.classifiers.multilabel.PSt;
import meka.classifiers.multilabel.RAkEL;
import meka.classifiers.multilabel.meta.BaggingML;
import meka.classifiers.multilabel.meta.DeepML;
import meka.classifiers.multilabel.meta.EnsembleML;
import meka.classifiers.multilabel.meta.MBR;
import meka.classifiers.multilabel.meta.RandomSubspaceML;
import meka.classifiers.multilabel.meta.SubsetMapper;
import utils.FilesOutput;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;

/**
 * Evaluation.java - Evaluation functionality.
 *
 * @author Jesse Read
 * @version March 2014
 */
public class ExperimentBatch{
    
    /**
     * Run experiment using command prompt args
     * @param args
     * @throws Exception 
     */
    public static void execute(String fileTrain, String fileTest, String outputDirectory, String numWindows) throws Exception {
            FilesOutput.createDirectory(outputDirectory);
            runExperiment(fileTrain, fileTest, Integer.parseInt(numWindows), outputDirectory);
    }
    
    public static void execute(int numWindows, String dataSetName) throws Exception {
            
            if(dataSetName.equals("sem_NP/4CRE-V2")){
                String fileTrain = "D:\\Google Drive\\datasets\\Datasets_Sem_NP\\4CRE-V2_train.arff";
                String fileTest = "D:\\Google Drive\\datasets\\Datasets_Sem_NP\\4CRE-V2_test.arff";
                String outputDirectory = "C:\\Users\\Joel\\Dropbox\\ASOC_2020\\MINAS-BR_ASOC_v2\\experiments\\batchMethods\\"+dataSetName+"\\";
                FilesOutput.createDirectory(outputDirectory);
                runExperiment(fileTrain, fileTest, numWindows, outputDirectory);
                
            }else if(dataSetName.equals("sem_NP/MOA-3C-5C-2D")){
                String fileTrain = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\MOA-3C-5C-2D_train.arff";
                String fileTest = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\MOA-3C-5C-2D_test.arff";
                String outputDirectory = "experiments/batchMethods/"+dataSetName+"/";
                runExperiment(fileTrain, fileTest, numWindows, outputDirectory);
                
            }else if(dataSetName.equals("sem_NP/MOA-5C-7C-2D")){
                String fileTrain = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\MOA-5C-7C-2D_train.arff";
                String fileTest = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\MOA-5C-7C-2D_test.arff";
                String outputDirectory = "experiments/batchMethods/"+dataSetName+"/";
                runExperiment(fileTrain, fileTest, numWindows, outputDirectory);
                
            }else if(dataSetName.equals("sem_NP/MOA-5C-7C-3D")){
                String fileTrain = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\MOA-5C-7C-3D_train.arff";
                String fileTest = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\MOA-5C-7C-3D_test.arff";
                String outputDirectory = "experiments/batchMethods/"+dataSetName+"/";
                runExperiment(fileTrain, fileTest, numWindows, outputDirectory);
                
            }else if(dataSetName.equals("sem_NP/nus-wide")){
                String fileTrain = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\nus-wide_modified_train.arff";
                String fileTest = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\nus-wide_modified_test.arff";
                String outputDirectory = "experiments/batchMethods/"+dataSetName+"/";
                runExperiment(fileTrain, fileTest, numWindows, outputDirectory);
            }else if(dataSetName.equals("sem_NP/Yeast")){
                String fileTrain = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\yeast-V2_modified_train.arff";
                String fileTest = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\yeast-V2_modified_test.arff";
                String outputDirectory = "experiments/batchMethods/"+dataSetName+"/";
                runExperiment(fileTrain, fileTest, numWindows, outputDirectory);
            }else if(dataSetName.equals("sem_NP/Scene")){
                String fileTrain = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\scene-V2_train.arff";
                String fileTest = "D:\\Google Drive\\datasets\\datasets_Sem_NP\\scene-V2_test.arff";
                String outputDirectory = "experiments/batchMethods/"+dataSetName+"/";
                runExperiment(fileTrain, fileTest, numWindows, outputDirectory);
            }
    }
    
    private static void runExperiment(String fileTrain, String fileTest, int numWindows, String outputDirectory) throws Exception{
        ArrayList<Evaluator> listEvaluators = new ArrayList<>();
        Instances trainSet = DataSetUtils.loadDataset(fileTrain);
        MLUtils.prepareData(trainSet);
        Instances testSet = DataSetUtils.loadDataset(fileTest);
        MLUtils.prepareData(testSet);
        float evaluationWindowsSizeAux = (float)testSet.size()/(float)numWindows;
        int evaluationWindowsSize = (int)Math.ceil(evaluationWindowsSizeAux);
        
        PSt pst = new PSt();
        String nameMethod = "PSt";
        listEvaluators.add(executeClassifier(pst,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
        CC cc = new CC();
        nameMethod = "CC";
        listEvaluators.add(executeClassifier(cc,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));

        MULAN homer = new MULAN();
        int partitions = trainSet.classIndex() > 2 ? 3 : 2;
        homer.setMethod("HOMER.Random."+partitions+".LabelPowerset");
        nameMethod = "HOMER";
        listEvaluators.add(executeClassifier(homer,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
        MULAN CLR = new MULAN();
        CLR.setMethod("CLR");
        nameMethod = "CLR";
        listEvaluators.add(executeClassifier(CLR,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
        MULAN BPMLL = new MULAN();
        BPMLL.setMethod("BPMLL");
        nameMethod = "BPMLL";
        listEvaluators.add(executeClassifier(BPMLL,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
        MULAN MLkNN = new MULAN();
        MLkNN.setMethod("MLkNN");
        nameMethod = "MLkNN";
        listEvaluators.add(executeClassifier(MLkNN,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
//        MULAN IBLR_ML = new MULAN();
//        IBLR_ML.setMethod("IBLR_ML");
//        nameMethod = "IBLR_ML";
//        listEvaluators.add(executeClassifier(IBLR_ML,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
//        DBPNN deepBackPropagation = new DBPNN();
//        nameMethod = "DBPNN";
//        listEvaluators.add(executeClassifier(deepBackPropagation,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
        FW fourClassPairwise = new FW();
        nameMethod = "FourClassPairwise";
        listEvaluators.add(executeClassifier(fourClassPairwise,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
        if(trainSet.classIndex() > 2){
            RAkEL rakel = new RAkEL();
            nameMethod = "RAkEL";
            listEvaluators.add(executeClassifier(rakel,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        }
        
        //Ensemble
        BaggingML baggingML = new BaggingML();
//        PSt pSt = new PSt();
//        pSt.setClassifier(new NaiveBayes());
//        baggingML.setClassifier(pSt);
        nameMethod = "BaggingML";
        listEvaluators.add(executeClassifier(baggingML,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
//        DeepML deepML= new DeepML();
////        PSt pSt_deepML = new PSt();
////        pSt_deepML.setClassifier(new NaiveBayes());
////        deepML.setClassifier(pSt_deepML);
//        nameMethod = "DeepML";
//        listEvaluators.add(executeClassifier(deepML,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
//        EnsembleML esembleML= new EnsembleML();
//        PSt pSt = new PSt();
//        pSt.setClassifier(new NaiveBayes());
//        esembleML.setClassifier(pSt);
//        nameMethod = "EnsembleML";
//        listEvaluators.add(executeClassifier(esembleML,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
        RandomSubspaceML randomSubspaceML= new RandomSubspaceML();
//        PSt pSt_deepML = new PSt();
//        pSt_deepML.setClassifier(new NaiveBayes());
//        deepML.setClassifier(pSt_deepML);
        nameMethod = "RandomSubspaceML";
        listEvaluators.add(executeClassifier(randomSubspaceML,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
         SubsetMapper subsetMapper = new SubsetMapper();
//        PSt pSt_deepML = new PSt();
//        pSt_deepML.setClassifier(new NaiveBayes());
//        deepML.setClassifier(pSt_deepML);
        nameMethod = "SubsetMapper";
        listEvaluators.add(executeClassifier(subsetMapper,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
        MBR mbr = new MBR();
//        PSt pSt_deepML = new PSt();
//        pSt_deepML.setClassifier(new NaiveBayes());
//        deepML.setClassifier(pSt_deepML);
        nameMethod = "MBR";
        listEvaluators.add(executeClassifier(mbr,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
        EvaluatorBR.writesAvgResults(listEvaluators, outputDirectory);
    }
    
    private static void runExperiment(String fileTrain, String fileTest, int numWindows, String nameMethod, String outputDirectory) throws Exception{
        ArrayList<Evaluator> listEvaluators = new ArrayList<>();
        Instances trainSet = DataSetUtils.loadDataset(fileTrain);
        MLUtils.prepareData(trainSet);
        Instances testSet = DataSetUtils.loadDataset(fileTest);
        MLUtils.prepareData(testSet);
        float evaluationWindowsSizeAux = (float)testSet.size()/(float)numWindows;
        int evaluationWindowsSize = (int)Math.ceil(evaluationWindowsSizeAux);
        
        if(nameMethod.equals("PSt")){
            PSt pst = new PSt();
            listEvaluators.add(executeClassifier(pst,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
            
        }else if(nameMethod.equals("CC")){
            CC cc = new CC();
            listEvaluators.add(executeClassifier(cc,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
            
        }else if(nameMethod.equals("HOMER")){
            MULAN homer = new MULAN();
            int partitions = trainSet.classIndex() > 2 ? 3 : 2;
            homer.setMethod("HOMER.Random."+partitions+".LabelPowerset");
            nameMethod = "HOMER";
            listEvaluators.add(executeClassifier(homer,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
            
        }else if(nameMethod.equals("CLR")){
            MULAN CLR = new MULAN();
            CLR.setMethod("CLR");
            listEvaluators.add(executeClassifier(CLR,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
            
        }else if(nameMethod.equals("BPMLL")){
            MULAN BPMLL = new MULAN();
            BPMLL.setMethod("BPMLL");
            listEvaluators.add(executeClassifier(BPMLL,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
            
        }else if(nameMethod.equals("MLkNN")){
                MULAN MLkNN = new MULAN();
                MLkNN.setMethod("MLkNN");
                listEvaluators.add(executeClassifier(MLkNN,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));

        }else if(nameMethod.equals("IBLR_ML")){
            MULAN IBLR_ML = new MULAN();
            IBLR_ML.setMethod("IBLR_ML");
            listEvaluators.add(executeClassifier(IBLR_ML,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
        }else if(nameMethod.equals("DBPNN")){
            DBPNN deepBackPropagation = new DBPNN();
            nameMethod = "DBPNN";
            listEvaluators.add(executeClassifier(deepBackPropagation,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
        }else if(nameMethod.equals("FourClassPairwise")){
            FW fourClassPairwise = new FW();
            listEvaluators.add(executeClassifier(fourClassPairwise,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
            
        }else if(nameMethod.equals("RAkEL")){
            if(trainSet.classIndex() > 2){
                RAkEL rakel = new RAkEL();
                listEvaluators.add(executeClassifier(rakel,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
            }
            
        }else if(nameMethod.equals("BaggingML")){
        //Ensemble
            BaggingML baggingML = new BaggingML();
    //        PSt pSt = new PSt();
    //        pSt.setClassifier(new NaiveBayes());
    //        baggingML.setClassifier(pSt);
            listEvaluators.add(executeClassifier(baggingML,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
            
        }else if(nameMethod.equals("DeepML")){
            DeepML deepML= new DeepML();
    //        PSt pSt_deepML = new PSt();
    //        pSt_deepML.setClassifier(new NaiveBayes());
    //        deepML.setClassifier(pSt_deepML);
            listEvaluators.add(executeClassifier(deepML,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        }else if(nameMethod.equals("EnsembleML")){
            EnsembleML esembleML= new EnsembleML();
            PSt pSt = new PSt();
            pSt.setClassifier(new NaiveBayes());
            esembleML.setClassifier(pSt);
            listEvaluators.add(executeClassifier(esembleML,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        }else if(nameMethod.equals("RandomSubspaceML")){
            RandomSubspaceML randomSubspaceML= new RandomSubspaceML();
    //        PSt pSt_deepML = new PSt();
    //        pSt_deepML.setClassifier(new NaiveBayes());
    //        deepML.setClassifier(pSt_deepML);
            nameMethod = "RandomSubspaceML";
            listEvaluators.add(executeClassifier(randomSubspaceML,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        
        }else if(nameMethod.equals("SubsetMapper")){
            SubsetMapper subsetMapper = new SubsetMapper();
   //        PSt pSt_deepML = new PSt();
   //        pSt_deepML.setClassifier(new NaiveBayes());
   //        deepML.setClassifier(pSt_deepML);
           nameMethod = "SubsetMapper";
           listEvaluators.add(executeClassifier(subsetMapper,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
           
        }else if(nameMethod.equals("MBR")){
            MBR mbr = new MBR();
    //        PSt pSt_deepML = new PSt();
    //        pSt_deepML.setClassifier(new NaiveBayes());
    //        deepML.setClassifier(pSt_deepML);
            listEvaluators.add(executeClassifier(mbr,nameMethod,trainSet,testSet,evaluationWindowsSize,outputDirectory));
        }
        
        EvaluatorBR.writesAvgResults(listEvaluators, outputDirectory);
    }

    /**
     * Execute classifiers individually
     *
     * @param mlClassifier
     * @param nameMethod
     * @param trainSet
     * @param testSet
     * @param evaluationWindowsSize
     * @param outputDirectory
     * @return 
     * @throws java.lang.Exception
     */
    private static Evaluator executeClassifier(MultiLabelClassifier mlClassifier, String nameMethod, 
        Instances trainSet, Instances testSet, int evaluationWindowsSize, String outputDirectory) throws Exception {
        String output = outputDirectory + "/" + nameMethod + "/";
        FilesOutput.createDirectory(output);
        // Load Instances from a file
        int numLabels = trainSet.classIndex();
        Set<String> knownClasses = DataSetUtils.getClassesConhecidas(trainSet, numLabels);
        Evaluator evaluator = new Evaluator(numLabels,knownClasses,nameMethod);

        // Train
        mlClassifier.buildClassifier(trainSet);

        // Threshold OPtion
        String top = "PCut1"; // default
        
        Result result = testClassifier(mlClassifier, testSet);
        double[][] pred = result.allPredictions();
        int[][] trueLabels = result.allTrueValues();

        // if PCut is specified we need the training data,
        // so that we can calibrate the threshold!
        String threshold = MLEvalUtils.getThreshold(result.predictions, trainSet, top);

        //thresholding
        double thresholdForEachClass[] = ThresholdUtils.thresholdStringToArray(threshold, trueLabels[0].length);
        int Ypred[][] = ThresholdUtils.threshold(pred, thresholdForEachClass);

        ArrayList<Set<String>> trueLabelsSet = DataSetUtils.predictionMatrixToSet(trueLabels);
        ArrayList<Set<String>> predSet = DataSetUtils.predictionMatrixToSet(Ypred);
        int window = 1;
        for (int i = 0; i < trueLabelsSet.size(); i++) {
//            System.out.println("Real: " +  trueLabelsSet.get(i) + "\t" + "Predicted: " + predSet.get(i).toString());
            evaluator.updateExampleBasedMeasures(predSet.get(i), trueLabelsSet.get(i));
            evaluator.updateLabelBasedMeasures(predSet.get(i), trueLabelsSet.get(i));

            if(i > 0 && i % evaluationWindowsSize == 0){
                System.out.println("###Evaluation window: " + window);
                evaluator.calculateWindowMeasures();
                window++;
            }
        }
        evaluator.calculateWindowMeasures();
//        FilesOutput.createDirectory(outputDirectory+"\\"+nameMethod+"\\");
        evaluator.writeMeasuresOverTime(output);
        evaluator.writesAvgResults(outputDirectory);
        return evaluator;
    }
    
    

    /**
     * TestClassifier - test classifier h on testSet
     *
     * @param	mlclassifier	a multi-dim. classifier, ALREADY BUILT
     * @param	testSet test data
     * @return	Result	with raw prediction data ONLY
     */
    private static Result testClassifier(MultiLabelClassifier mlclassifier, Instances testSet) throws Exception {

        int numLabels = testSet.classIndex();
        Result result = new Result(testSet.numInstances(), numLabels);

        for (int i = 0, c = 0; i < testSet.numInstances(); i++) {
            // No cheating allowed; clear all class information
            Instance instance = (Instance)testSet.instance(i).copy();
            
            for (int v = 0; v < testSet.classIndex(); v++) {
                instance.setValue(v, 0.0);
            }

            // Get and store ranking
            double trueLabels[] = null;
            try{
                trueLabels = mlclassifier.distributionForInstance(instance);
            }catch(Exception e){
                e.printStackTrace();
                System.exit(0);
            }

            // Store the result
            result.addResult(trueLabels, testSet.instance(i));
        }
        return result;
    }

}
