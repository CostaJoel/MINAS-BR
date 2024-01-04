package main;


import baselines.batch.ExperimentBatch;
import baselines.upperbound.ExperimentUpperBound;
import br.Model;
import br.OfflinePhase;
import br.OnlinePhase;
import com.yahoo.labs.samoa.instances.Instance;
import dataSource.DataSetUtils;
import evaluate.Evaluator;
import evaluate.EvaluatorBR;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import moa.streams.MultiTargetArffFileStream;
import utils.FilesOutput;
import weka.core.Instances;


public class Main {

    public static void main(String[] args) throws Exception {
          
        //****************General*********************
        String dataSetName = args[1];
        String trainPath = args[2];
        String testPath = args[3];
        int L = Integer.parseInt(args[4]);
        String outputDirecotory = "";
        if(args[0].equals("-p")){
            outputDirecotory = args[5];
            
            experimentsParameters(dataSetName,
                trainPath,
                testPath,
                L,
                outputDirecotory);
        }else if (args[0].equals("-m")){
            double k_ini = Double.parseDouble(args[5]);
            String theta = "" + (int) Math.ceil(Double.parseDouble(args[6]) * Double.parseDouble(args[7]));
            String omega = args[7];
            outputDirecotory = args[8] + "/" + dataSetName;
            
            experimentsMethods(trainPath, 
                testPath, 
                outputDirecotory,
                L, 
                k_ini,
                theta, 
                omega,
                "1.1",
                "kmeans+leader",
                "JI");
        }else if (args[0].equals("-b")){
            outputDirecotory = args[5] + "/batch/"+dataSetName+"/";
            ExperimentBatch.execute(trainPath, testPath, outputDirecotory, "50");
        }else if(args[0].equals("-o")){
            outputDirecotory = args[5] + "/upperBound/"+dataSetName+"/";
            ExperimentUpperBound.execute(trainPath, testPath, outputDirecotory, 50);
        }
        
//        //****************MOA-3C*********************
//        String dataSetName = "MOA-3C_teste_updating";
//        String trainPath = "/home/joel/datasets/datasets_sinteticos/MOA-3C-5C-2D/MOA-3C-5C-2D-train.arff";
//        String testPath = "/home/joel/datasets/datasets_sinteticos/MOA-3C-5C-2D/MOA-3C-5C-2D-test.arff";
//        String outputDirecotory = "/home/joel/MINAS-BR_ASOC/results_fev/"+dataSetName+"/";
//        double k_ini = 0.1;
//        String theta = "20";
//        String omega = "200";
//        int L = 5;
        //*****************************************
        
//        //****************MOA1*********************
//        String dataSetName = "minas-br_MOA1";
////        String dataSetName = "MOA1_kini=0.001";
//        String trainPath = "/home/joel/datasets/datasets_sinteticos/MOA-5C-7C-2D/MOA-5C-7C-2D-train.arff";
//        String testPath = "/home/joel/datasets/datasets_sinteticos/MOA-5C-7C-2D/MOA-5C-7C-2D-test.arff";
//        String outputDirecotory = "/home/joel/resultsAsoc2021/results_to_plot/MOA1/"+dataSetName+"/";
//        double k_ini = 0.01;
////        double k_ini = 0.001;
//        double theta = 0.75;
//        String omega = "2000";
//        String theta_param = "" + (int)Math.ceil(Double.parseDouble(omega) * theta);
//        int L = 7;
//        //*****************************************

//        //****************yeast*********************
//        String dataSetName = "batch_yeast_original";
//        String trainPath = "/home/joel/datasets/datasets_reais/Yeast/yeast_original_train.arff";
//        String testPath = "/home/joel/datasets/datasets_reais/Yeast/yeast_original_test.arff";
//        String outputDirectory = "/home/joel/resultsAsoc2021/baseline_methods/batchMethods/"+dataSetName+"/";
//        double k_ini = 0.25;
////        double k_ini = 0.001;
//        String theta = "20";
//        String omega = "200";
//        int L = 14;
////        //*****************************************
        
        //****************mediamill*********************
//        String dataSetName = "mediamill_original_validation";
////        String dataSetName = "MOA1_kini=0.001";
//        String trainPath = "/home/joel/datasets/datasets_reais/mediamill/mediamill_original_train.arff";
//        String testPath = "/home/joel/datasets/datasets_reais/mediamill/mediamill_original_test.arff";
//        String outputDirecotory = "/home/joel/resultsAsoc2021/validation/"+dataSetName+"/";
//        double k_ini = 0.1;
////        double k_ini = 0.001;
//        String theta = "200";
//        String omega = "2000";
//        int L = 101;
//        //*****************************************

        //****************scene*********************
//        String dataSetName = "batch_scene_modified";
////        String dataSetName = "MOA1_kini=0.001";
//        String trainPath = "/home/joel/datasets/datasets_modified/scene/scene-V2_train.arff";
//        String testPath = "/home/joel/datasets/datasets_modified/scene/scene-V2_test.arff";
//        String outputDirectory = "/home/joel/resultsAsoc2021/baseline_methods/batchMethods/"+dataSetName+"/";
//        double k_ini = 0.1;
////        double k_ini = 0.001;
//        String theta = "20";
//        String omega = "200";
//        int L = 6;
//        //*****************************************
        //****************scene modified*********************
//        String dataSetName = "upper_scene_original";
////        String dataSetName = "MOA1_kini=0.001";
//        String trainPath = "/home/joel/datasets/datasets_reais/Scene/scene_original_train.arff";
//        String testPath = "/home/joel/datasets/datasets_reais/Scene/scene_original_test.arff";
//        String outputDirectory = "/home/joel/resultsAsoc2021/baseline_methods/upperBoundMethods/"+dataSetName+"/";
//        double k_ini = 0.1;
////        double k_ini = 0.001;
//        String theta = "20";
//        String omega = "200";
//        int L = 6;
//        //*****************************************

////        //****************MOA2*********************
//        String dataSetName = "MOA2";
//        String trainPath = "/home/joel/Documents/datasets/datasets_sinteticos/4CRE-V2/4CRE-V2-train.arff";
//        String testPath = "/home/joel/Documents/datasets/datasets_sinteticos/4CRE-V2/4CRE-V2-test.arff";
//        String outputDirecotory = "/home/joel/results_jan2021/"+dataSetName+"/";
//        double k_ini = 0.01;
//        String theta = "200";
//        String omega = "200";
//        int L = 4;
//        //*****************************************
        
//        experimentsMethods(trainPath, 
//                testPath, 
//                outputDirecotory,
//                L, 
//                k_ini,
//                theta_param, 
//                omega,
//                "1",
//                "kmeans+leader",
//                "JI");
//        
//        experimentsParameters(dataSetName,
//                trainPath,
//                testPath,
//                L,
//                outputDirecotory);
//        ExperimentBatch.execute(trainPath, testPath, outputDirectory, "50");
//        String dataSetName = "upper_nus-wide_original";
//        String trainPath = "/home/joel/datasets/datasets_reais/Nuswide_cVLADplus/nus-wide_original_train.arff";
//        String testPath = "/home/joel/datasets/datasets_reais/Nuswide_cVLADplus/nus-wide_original_test.arff";
//        String outputDirecotory = "/home/joel/resultsAsoc2021/baseline_methods/upperBoundMethods/"+dataSetName+"/";
//        int L = 81; 
//        ExperimentUpperBound.execute(trainPath, testPath, outputDirecotory, 50);
    }
    
    private static void experimentsParameters(String dataSetName,
            String trainPath, 
            String testPath,
            int L,
            String outputDirectory) throws IOException, Exception {
        
        ArrayList<Instance> train = new ArrayList<Instance>();
        ArrayList<Instance> test = new ArrayList<Instance>();
        
        MultiTargetArffFileStream file = new MultiTargetArffFileStream(trainPath, String.valueOf(L));
        file.prepareForUse();
        while(file.hasMoreInstances()){
            train.add(file.nextInstance().getData());
        }
        file.restart();
        
        file = new MultiTargetArffFileStream(testPath, String.valueOf(L));
        file.prepareForUse();
        
        while(file.hasMoreInstances()){
            test.add(file.nextInstance().getData());
        }
        file.restart();
//        double[] theta = {1};
//        int[] omega = {2000};
//        double[] f = {1.1};
//        double[] k_ini = {0.01,0.001};
        double[] theta = {0.1,0.25,0.5,0.75,1};
        int[] omega = {200, 500, 1000, 2000};
        double[] f = {1.1};
        double[] k_ini = {0.01, 0.05, 0.1, 0.25};

        outputDirectory = outputDirectory + "/parameters_sensitivity/"+dataSetName+"/";
        FilesOutput.createDirectory(outputDirectory);
        
        
        FileWriter fileResults = new FileWriter(new File(outputDirectory + "/fullResults.csv"), false);
        fileResults.write("shortTermMemoryLimit,windowsSize,threshold,k_ini,F1M,precision,recall,subsetAccuracy,NP,unk,unkRemoved\n");
        fileResults.close();
        
        for (int i = 0; i < theta.length; i++) {
            for (int j = 0; j < omega.length; j++) {
                if(theta[i] <= omega[j]){
                    for (int k = 0; k < f.length; k++) {
                        for (int l = 0; l < k_ini.length; l++) {
                            String dir = outputDirectory + theta[i] + "_" + omega[j] + "_" + f[k]+ "_" + k_ini[l] +"/";
                            System.out.println("***********"+theta[i] + "_" + omega[j] + "_" + f[k]+ "_" + k_ini[l]+"******************");
                            FilesOutput.createDirectory(dir);
                            EvaluatorBR avMINAS = MINAS_BR(train,
                                    test,
                                    L, 
                                    k_ini[l],
                                    (int) theta[i]*omega[j],
                                    omega[j], 
                                    f[k],
                                    dir);
                            
                            fileResults = new FileWriter(new File(outputDirectory + "/fullResults.csv"), true);
                            
                            fileResults.write(theta[i]+","+
                                    omega[j]+","+
                                    f[k] + ","+
                                    k_ini[l]+","+
                                    avMINAS.getAvgF1M() + ","+
                                    avMINAS.getAvgPr()+ ","+
                                    avMINAS.getAvgRe() + ","+
                                    avMINAS.getAvgSA() + "," +
                                    avMINAS.getQtdeNP() + "," +
                                    avMINAS.getUnk().stream().mapToInt(p -> p).sum() + ","+
                                    avMINAS.getRemovedUnk().stream().mapToInt(p -> p).sum() + "\n");
                            fileResults.close();
                        }
                    }
                }
            }
        }
    }
    
    public static EvaluatorBR MINAS_BR(ArrayList<Instance> train,
            ArrayList<Instance> test,
            int L, 
            double k_ini, 
            int theta, 
            int omega, 
            double f, 
            String outputDirectory) throws IOException, Exception {
        
//        Instances D_ = DataSetUtils.dataFileToInstance(dataSetPath);
//        int L = D_.classIndex();
//        int streamSize = D_.numInstances();
//        int L = 81;
//        int streamSize = 269648;
        
//        MultiTargetArffFileStream stream = new MultiTargetArffFileStream(dataSetPath, String.valueOf(L));
//        stream.prepareForUse();
//        ArrayList<Instance> train = new ArrayList<Instance>();
//        ArrayList<Instance> test = new ArrayList<Instance>();
//        slipTrainTest(train, test, stream, streamSize, 0.1);
//        int[] dist = DataSetUtils.getLabelsDistribution(train);
//        int[] distTest = DataSetUtils.getLabelsDistribution(train);
//        float[] windowsCardinalities = DataSetUtils.getWindowsCardinalities(test, evaluationWindowSize, L);

        //Create output files
        FileWriter filePredictions = new FileWriter(new File(outputDirectory + "/predictionsInfo.csv"), false); //Armazena informações da fase online
        filePredictions.write("timestamp;actual;predicted" + "\n");
        FileWriter fileOff = new FileWriter(new File(outputDirectory + "/faseOfflineInfo.txt"), false); //Armazena informações da fase online
        FileWriter fileOut = new FileWriter(new File(outputDirectory + "/results.txt"), false); //Armazena informações da fase de treinamento
        
        int evaluationWindowSize = (int) Math.ceil(test.size()/50.0);
        
        OfflinePhase treino = new OfflinePhase(train, k_ini, fileOff, outputDirectory);
        Model model = treino.getModel();
        model.setEvaluationWindowSize(evaluationWindowSize);
        model.writeCurrentCardinality(1, outputDirectory);
        
        fileOff.write("Known Classes: " + model.getAllLabel().size() + "\n");
         fileOff.write("Train label cardinality: " + model.getCurrentCardinality() + "\n");
//        fileOff.write("Windows label cardinality: " + Arrays.toString(windowsCardinalities) + "\n");
        fileOff.write("Number of examples: " + (train.size()+test.size()) + "\n");
        fileOff.write("Number of attributes: " + train.get(0).numInputAttributes() +"\n");
        
        EvaluatorBR av = new EvaluatorBR(L, model.getModel().keySet(), "MINAS-BR"); 
        OnlinePhase onlinePhase = new OnlinePhase(theta, f, outputDirectory, fileOut, "kmeans+leader");
        
        //Classification phase
        for (int i = 0; i < test.size(); i++) {
            onlinePhase.incrementarTimeStamp();
            System.out.println("Timestamp: " + onlinePhase.getTimestamp());
            onlinePhase.classify(model, av, test.get(i),filePredictions);
            
            //for each model deletes the micro-clusters wich have not been used
            if((onlinePhase.getTimestamp()%omega) == 0){
                model.resetMtxLabelFrequencies(omega);
                model.updateCardinality(omega);
                model.writeCurrentCardinality(onlinePhase.getTimestamp(), outputDirectory);
                model.writeBayesRulesElements(onlinePhase.getTimestamp(), outputDirectory);
                model.clearSortTimeMemory(omega, onlinePhase.getTimestamp(),fileOut, false);
                onlinePhase.removeOldMicroClusters(omega, model, fileOut);
            }
            if((onlinePhase.getTimestamp()%evaluationWindowSize) == 0){
//                model.writeBayesRulesElements(onlinePhase.getTimestamp(), outputDirectory);
//                model.writeCurrentCardinality(onlinePhase.getTimestamp(), outputDirectory);
                model.associatesNPs(evaluationWindowSize, onlinePhase.getTimestamp(), "JI");
//                av.getDeletedExamples().add(model.getShortTimeMemory().getQtdeExDeleted());
                av.updateExampleBasedMeasure(model, evaluationWindowSize);
                av.updateLabelBasedMeasure(model, evaluationWindowSize);
            }
            if(i == test.size()-1 && (onlinePhase.getTimestamp()%evaluationWindowSize)>0){
                model.associatesNPs(evaluationWindowSize, onlinePhase.getTimestamp(), "JI");
                av.updateExampleBasedMeasure(model, evaluationWindowSize);
                av.updateLabelBasedMeasure(model, evaluationWindowSize);
            }
        }
        onlinePhase.getExtInfo().close();
        av.setQtdeNP(model.getNPs().size());
        av.writeMeasuresOverTime(outputDirectory);
        av.writeConceptEvolutionNP(model, outputDirectory);
        fileOut.close();
        filePredictions.close();
        fileOff.close();
        System.out.println("Number of examples sent to short-time-memory = " + onlinePhase.getExShortTimeMem());
        model.getPnInfo().write("Number of examples sent to short-time-memory = " + onlinePhase.getExShortTimeMem() + "\n");
        System.out.println("Number of examples removed from short-time-memory = " + model.getShortTimeMemory().getQtdeExDeleted());
        model.getPnInfo().write("Number of examples removed from short-time-memory = " + model.getShortTimeMemory().getQtdeExDeleted()+ "\n");
        System.out.println("Number of NPs = " + model.getNPs().size());
        model.getPnInfo().write("Number of NPs = " + model.getNPs().size()+ "\n");
        model.getPnInfo().close();
        return av;
    }
    
    

    /**
     * Experiments to compare MINAS-BR with others methods
     *
     * @param dataSetName
     * @param dataSetPath
     * @param omega window size
     * @param theta limit of short-term memory 
     * @param f
     * @param algOn
     * @param evMetric
     * @param outputDirectory
     * @throws Exception
     */
    public static void experimentsMethods(String trainPath, 
            String testPath,
            String outputDirectory,
            int L, 
            double k_ini,
            String theta,
            String omega, 
            String f,
            String algOn,
            String evMetric) throws Exception {
        
       FilesOutput.createDirectory(outputDirectory);
       ArrayList<Instance> train = new ArrayList<Instance>();
       ArrayList<Instance> test = new ArrayList<Instance>();
        
        MultiTargetArffFileStream file = new MultiTargetArffFileStream(trainPath, String.valueOf(L));
        file.prepareForUse();
        while(file.hasMoreInstances()){
            train.add(file.nextInstance().getData());
        }
        file.restart();
        
        file = new MultiTargetArffFileStream(testPath, String.valueOf(L));
        file.prepareForUse();
        
        while(file.hasMoreInstances()){
            test.add(file.nextInstance().getData());
        }
        file.restart();
        
        ArrayList<Instance> aux = new ArrayList<>();
        aux.addAll(train);
        aux.addAll(test);
        float cardinalityTrain = DataSetUtils.getCardinality(train, L);
        float labelCardinality = DataSetUtils.getCardinality(aux, L);
        FileWriter DsInfos = new FileWriter(new File(outputDirectory + "/dataSetInfo.txt"), false);
        

        DsInfos.write("Train label cardinality: " + cardinalityTrain + "\n");
        DsInfos.write("General label cardinality: " + labelCardinality + "\n");
//        DsInfos.write("Windows label cardinality: " + Arrays.toString(windowsCardinalities) + "\n");
        DsInfos.write("Number of examples: " + train.size()+test.size() + "\n");
        DsInfos.write("Number of attributes: " + train.get(0).numInputAttributes() +"\n");
        DsInfos.close();
        
        ArrayList<Evaluator> av = new ArrayList<Evaluator>();
        av.add(MINAS_BR(train,
                test,
                L, 
                k_ini,
                Integer.valueOf(theta), 
                Integer.valueOf(omega), 
                Double.valueOf(f),
                outputDirectory)
        );

        EvaluatorBR.writesAvgResults(av,outputDirectory);
        Evaluator.writeMeasuresOverTime(av, outputDirectory);
    }
    

    private static void removeClasses(String dataSetPath) throws IOException {
        MultiTargetArffFileStream stream = new MultiTargetArffFileStream(dataSetPath, "1");
        stream.prepareForUse();
        
        FileWriter dataSetFile = new FileWriter(new File(dataSetPath.replace(".arff", "-CE.arff")), false);
        
        //write the file's header
        dataSetFile.write("@relation \n");
        dataSetFile.write("@attribute att1 numeric \n");
        dataSetFile.write("@attribute att2 numeric \n");
        dataSetFile.write("@attribute class \n");
        dataSetFile.write("\n");
        dataSetFile.write("@data\n");
        
        int cont1 = 1;
        int cont2 = 1;
        while(cont1 <= 50000){
            Instance inst = stream.nextInstance().getData();
            System.out.println(inst.value(2) );
            if(inst.value(2) != 1.0 && inst.value(2) != 2.0){
                for (int i = 0; i < inst.numAttributes(); i++) {
                    dataSetFile.write(inst.value(i) + ",");
                }
                dataSetFile.write("\n");
            }
            cont1++;
            cont2++;
        }
        while(cont2 <= 100000){
            Instance inst = stream.nextInstance().getData();
            System.out.println(inst.value(2) );
            if(inst.value(2) != 2){
                for (int i = 0; i < inst.numAttributes(); i++) {
                    dataSetFile.write(inst.value(i) + ",");
                }
                dataSetFile.write("\n");
            }
            cont1++;
            cont2++;
        }
        while(stream.hasMoreInstances()){
            Instance inst = stream.nextInstance().getData();
            for (int i = 0; i < inst.numAttributes(); i++) {
                dataSetFile.write(inst.value(i) + ",");
            }
            dataSetFile.write("\n");
        }
        dataSetFile.close();
    }
    
}
