/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package br;

import NoveltyDetection.ClustreamKernelMOAModified;
import NoveltyDetection.KMeansMOAModified;
import NoveltyDetection.MicroCluster;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import dataSource.DataSetUtils;
import evaluate.EvaluatorBR;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import moa.cluster.CFCluster;
import moa.cluster.Clustering;
import moa.clusterers.KMeans;
import utils.Voting;

public class OnlinePhase {

    private int lastCheck = 0;			// last time the clustering algorithm was executed
    private final double threshold;
    private int qtdeExExcluidos = 0;
    private int exShortTimeMem = 0;
    private ArrayList<Integer> timeStampExtension = new ArrayList<>();
    private ArrayList<Integer> timeStampNP = new ArrayList<>();
    private FileWriter extInfo;
    private String algOnl;
    private int timestamp;
    private int shortTermMemoryLimit;          // minimum number of examples in the unknown memory to execute the ND procedure
    private String outputDirectory;
    private FileWriter fileOn;
    private ArrayList<MicroClusterBR> sleepMemory;
    
    public OnlinePhase(int theta, double threshold, String outputDirectory, FileWriter fileOn, String algOn) throws Exception {
        this.algOnl = algOn;
        this.shortTermMemoryLimit = theta;
        this.outputDirectory = outputDirectory;
        this.fileOn = fileOn;
        this.threshold = threshold;
        extInfo = new FileWriter(new File(outputDirectory+"/extInfo.txt"),false);
        sleepMemory = new ArrayList<>();
    }
    
    /**
     * Cluster with k-means
     *
     * @param numMClusters
     * @param dataSet
     * @param exampleCluster
     * @return
     * @throws NumberFormatException
     * @throws IOException
     */
    public ArrayList<MicroClusterBR> createModelKMeansOnline(int numMClusters, ArrayList<Instance> dataSet, int[] exampleCluster) throws NumberFormatException, IOException {
        ArrayList<MicroClusterBR> modelSet = new ArrayList<>();
        List<ClustreamKernelMOAModified> examples = new LinkedList<>();

        //Adicionando os exemplos ao algoritmo
        for (int k = 0; k < dataSet.size(); k++) {
            double[] data = Arrays.copyOfRange(dataSet.get(k).toDoubleArray(), dataSet.get(k).numOutputAttributes(), dataSet.get(k).numAttributes());
            Instance inst = new DenseInstance(1, data);
            examples.add(new ClustreamKernelMOAModified(inst, dataSet.get(k).numAttributes() - dataSet.get(k).numOutputAttributes(), k));
        }

        //********* K-Means ***********************
        //generate initial centers aleatory
        ClustreamKernelMOAModified[] centrosIni = new ClustreamKernelMOAModified[numMClusters];
        int nroaleatorio;
        List<Integer> numeros = new ArrayList<>();
        for (int c = 0; c < examples.size(); c++) {
            numeros.add(c);
        }
        Collections.shuffle(numeros);
        for (int i = 0; i < numMClusters; i++) {
            nroaleatorio = numeros.get(i);
            centrosIni[i] = examples.get(nroaleatorio);
        }

        //execution of the KMeans  
        Clustering centers;
        centers = KMeans.kMeans(centrosIni, examples);

        //*********results     
        // transform the results of kmeans in a data structure used by MINAS
        CFCluster[] res = new CFCluster[centers.size()];
        for (int j = 0; j < examples.size(); j++) {
            // Find closest kMeans cluster
            double minDistance = Double.MAX_VALUE;
            int closestCluster = 0;
            for (int i = 0; i < centers.size(); i++) {
                double distance = KMeansMOAModified.distance(centers.get(i).getCenter(), examples.get(j).getCenter());
                if (distance < minDistance) {
                    closestCluster = i;
                    minDistance = distance;
                }
            }

            // add to the cluster
            if (res[closestCluster] == null) {
                res[closestCluster] = (CFCluster) examples.get(j).copy();
            } else {
                res[closestCluster].add(examples.get(j));
            }
            exampleCluster[j] = closestCluster;
        }

        Clustering micros;
        micros = new Clustering(res);

        //*********remove micro-cluster with few examples
        ArrayList<ArrayList<Integer>> mapClustersExamples = new ArrayList<>();
        for (int a = 0; a < centrosIni.length; a++) {
            mapClustersExamples.add(new ArrayList<Integer>());
        }
        for (int g = 0; g < exampleCluster.length; g++) {
            mapClustersExamples.get(exampleCluster[g]).add(g);
        }

        int value;
        for (int i = 0; i < micros.size(); i++) {
            //remove micro-cluster with less than 3 examples
            if (micros.get(i) != null) {
                if (((ClustreamKernelMOAModified) micros.get(i)).getWeight() < 3) {
                    value = -1;
                } else {
                    value = i;
                }

                for (int j = 0; j < mapClustersExamples.get(i).size(); j++) {
                    exampleCluster[mapClustersExamples.get(i).get(j)] = value;
                }
                if (((ClustreamKernelMOAModified) micros.get(i)).getWeight() < 3) {
                    micros.remove(i);
                    mapClustersExamples.remove(i);
                    i--;
                }
            } else {
                micros.remove(i);
                mapClustersExamples.remove(i);
                i--;
            }
        }

        MicroClusterBR model_tmp;
        for (int w = 0; w < numMClusters; w++) {
            if ((micros.get(w) != null)) {
                model_tmp = new MicroClusterBR(new MicroCluster((ClustreamKernelMOAModified) micros.get(w), "", "normal", timestamp));
                modelSet.add(model_tmp);
            }
        }
        return modelSet;
    }

    /**
     * Cluster with clustream
     *
     * @param dataSet exemplos da memória temporária
     * @param numMClusters número de micro-grupos desejado
     * @param cl rótulo
     * @param exampleCluster exemplos que serão removidos da memoria temporaria
     * @return micro-grupos
     * @throws NumberFormatException
     * @throws IOException
     */
//    public ArrayList<MicroCluster> criamodeloCluStreamOnline(ArrayList<Instance> dataSet, int numMClusters, int[] exampleCluster) throws NumberFormatException, IOException {
//        ArrayList<MicroCluster> conjModelos = new ArrayList<MicroCluster>();
//        ClustreamOfflineBR jc = new ClustreamOfflineBR();
//
//        //execute the clustream
//        boolean executeKMeans = false;
//
//        Clustering micros = jc.CluStream(dataSet, this.nLabels, numMClusters, flagMicro, executeKMeans);
//
//        if (micros.size() > 0) {
//            if (flagMicro) {
//                int gruposTemp[] = jc.getClusterExamples();
//                for (int i = 0; i < gruposTemp.length; i++) {
//                    exampleCluster[i] = gruposTemp[i];
//                }
//            } else {
//                int temp[] = jc.getClusteringResults();
//                for (int g = 0; g < jc.getClusteringResults().length; g++) {
//                    exampleCluster[g] = temp[g];
//                }
//            }
//        }
//        for (int w = 0; w < micros.size(); w++) {
//            MicroCluster mdtemp = new MicroCluster((ClustreamKernelMOAModified) micros.get(w), "", "normal", timestamp);
//            // add the temporary model to the decision model 
//            conjModelos.add(mdtemp);
//        }
//        return conjModelos;
//    }
//    
    /**
     * Envia os micro-grupos que não receberam exemplos na última janela para uma memória temporária
     */
//    public void putClusterMemorySleep() throws IOException {
//        for (int i = 0; i < getModelo().size(); i++) {
//            if (getModelo().get(i).getTime() < (timestamp - windowSize)) {
//                SleepModeMem.add(getModelo().get(i));
//                System.out.println("Micro-Grupo Removido: " + i +" classes: " + getModelo().get(i).getLabelClass() + " categoria: " +  getModelo().get(i).getCategory());
//                this.fileOut.write("Micro-Grupo Removido: " + i +" classes: " + getModelo().get(i).getLabelClass() + " categoria: " +  getModelo().get(i).getCategory());
//                this.fileOut.write("\n");
//                getModelo().remove(i);
//                i--;
//            }
//        }
//    }
    
    
    
    /**
     * Deletes micro-clusters which have not been used for a time period
     * @param windowSize
     * @param modelo
     * @param fileOut
     * @throws IOException 
     */
    public void putClusterMemorySleep(int windowSize, ArrayList<MicroClusterBR> model, FileWriter fileOut) throws IOException {
        ArrayList<MicroClusterBR> listaMicro = new ArrayList<>();
        this.fileOn.write("Tamanho do Modelo: " + model.size());
        for (int i = 0; i < model.size(); i++) {
            if (model.get(i).getMicroCluster().getTime() < (timestamp - (windowSize))) {
                listaMicro.add(model.get(i));
                getSleepMemory().add(model.get(i));
                fileOut.write("Micro-Grupo Removido: " + i +" classes: " + model.get(i).getMicroCluster().getLabelClass() + " categoria: " +  model.get(i).getMicroCluster().getCategory());
                fileOut.write("\n");
                model.remove(i);
                i--;
            }
        }
        try{
            fileOn.write("Timestamp: " + this.timestamp + " - Micro grupos removidos: " + listaMicro.size() + " - Tamanho modelo ["+model.get(0).getMicroCluster().getLabelClass()+"]:" + model.size() + "\n");
            System.out.println("Timestamp: " + this.timestamp + " - Micro grupos removidos: " + listaMicro.size() + " - Tamanho modelo ["+model.get(0).getMicroCluster().getLabelClass()+"]:" + model.size());
        }catch(Exception e){
            
        }
        }
    
    public void plotPeriodicMicroClustersGraph(){
        
    }
    
     /**
     * @return the timestamp
     */
    public int getTimestamp() {
        return timestamp;
    }

    public void incrementarTimeStamp() {
        this.timestamp = this.timestamp + 1;
    }

    /**
     * @return the algOnl
     */
    public String getAlgOnl() {
        return algOnl;
    }

    /**
     * @param algOnl the algOnl to set
     */
    public void setAlgOnl(String algOnl) {
        this.algOnl = algOnl;
    }


    /**
     * @return the shortTimeMemory
     */
//    public ShortTimeMemory getShortTimeMemory() {
//        return shortTimeMemory;
//    }
//
//    /**
//     * @param shortTimeMemory the shortTimeMemory to set
//     */
//    public void setShortTimeMemory(ShortTimeMemory shortTimeMemory) {
//        this.shortTimeMemory = shortTimeMemory;
//    }

    /**
     * @return the shortTermMemoryLimit
     */
    public int getShortTermMemoryLimit() {
        return shortTermMemoryLimit;
    }

    /**
     * @param shortTermMemoryLimit the shortTermMemoryLimit to set
     */
    public void setShortTermMemoryLimit(int shortTermMemoryLimit) {
        this.shortTermMemoryLimit = shortTermMemoryLimit;
    }

    /**
     * @return the outputDirectory
     */
    public String getOutputDirectory() {
        return outputDirectory;
    }

    /**
     * @param outputDirectory the outputDirectory to set
     */
    public void setOutputDirectory(String outputDirectory) {
        this.outputDirectory = outputDirectory;
    }

    /**
     * @return the fileOn
     */
    public FileWriter getFileOn() {
        return fileOn;
    }

    /**
     * @param fileOn the fileOn to set
     */
    public void setFileOn(FileWriter fileOn) {
        this.fileOn = fileOn;
    }

    /**
     * @return the sleepMemory
     */
    public ArrayList<MicroClusterBR> getSleepMemory() {
        return sleepMemory;
    }

    
    
    /**
     * Classifies or rejects new examples
     * @param model
     * @param av
     * @param data
     * @param fileOut
     * @throws IOException 
     */
    public void classify(Model model, EvaluatorBR av, Instance data, FileWriter filePredictions) throws IOException {
        Set<String> trueLabels = DataSetUtils.getLabelSet(data); //get true labels
        // filePredictions.write(this.getTimestamp() + ";" + labels.toString()+";");
        String information = "Ex: " + this.getTimestamp() + "\t True Labels: " + trueLabels.toString() + "\t MINAS-BR Prediction: ";
        model.verifyConceptEvolution(trueLabels, this.timestamp);

        ArrayList<Voting> voting = model.getClosestMicroClusters(data, 1);
        //An example is consider unknown when it is outside all of micro-clusters' models
        if (!voting.isEmpty()) {
            //classifies
            System.out.println("Classify");
            Set<String> predictedLabels = model.bayesRuleToClassify(voting, data, this.timestamp);
            model.addPrediction(trueLabels, predictedLabels);
            model.incrementNumerOfObservedExamples();
            filePredictions.write(information + predictedLabels.toString() + "\n");
        } else {
            //rejects
             System.out.println("Reject");
            model.addPrediction(trueLabels, null);
            filePredictions.write(information + "unknwon"+"\n");
            exShortTimeMem++;
            model.getShortTimeMemory().add(data, this.getTimestamp());
            
            //If short-term memory reach its limit, then NP procedure
            if ((model.getShortTimeMemory().size() >= this.getShortTermMemoryLimit()) && 
                    (this.getLastCheck() + this.getShortTermMemoryLimit() < this.getTimestamp())) {
                System.out.println("************Novelty Detection Phase***********");
                noveltyPatternProcedure(model, av, filePredictions);
            }
        }
    }
    
    /**
     * Seleciona os rótulos mais relevantes de acordo com a menor distância
     *
     * @param voting lista com as informações dos micro-grupos mais próximos
     * @param cardinality cardinalidade atual
     * @return vetor de bipartições
     */
    public Set<String> thresholding(ArrayList<Voting> voting, int cardinality) {
        Collections.sort(voting); //Ordenando da menor distância para a maior
        Set<String> Z = new HashSet<String>();
        if(voting.size() <= cardinality){
            for (int i = 0; i < voting.size(); i++) {
                Z.add(voting.get(i).getlabel());
            }
        }else{
            for (int i = 0; i < cardinality; i++) {
                Z.add(voting.get(i).getlabel());
            }
        }
        return Z;
    }

    /**
     * Detects if the new micro-clusters are extensions or NPs.
     *
     * @param model
     * @param av
     * @param noveltyPatterns lista com os NPs
     * @param fileOut arquivo para a saida dos resultados
     * @throws IOException
     */
    private void noveltyPatternProcedure(Model model, EvaluatorBR ev, FileWriter filePredictions) throws IOException {
        int numMinExCluster = 3;
        this.setLastCheck(this.getTimestamp());
        String textoArq = "";
        
        //vector for mapping examples to be removed from short-time-memory because they were used to create a new valid micro-cluster                
        int[] removeExamples = new int[model.getShortTimeMemory().size()];
        
        //temporary new micro-cluster formed by unknown examples
        HashMap<Integer, MicroClusterBR> modelUnk = this.createModelKMeansLeader(model, 
                removeExamples,
                this.getTimestamp()
        );

        
        //for each candidate micro-cluster
        for (Map.Entry<Integer, MicroClusterBR> entry : modelUnk.entrySet()) {
            int indexMC = entry.getKey();
            MicroClusterBR currentEvaluatedMC = entry.getValue();
            List<Entry<Integer, Instance>> toClassify = new ArrayList<>(); //stores examples which will form the new micro-clusters to classify them afterwards
            ArrayList<MicroClusterBR> extModels = new ArrayList<>();
            ArrayList<MicroClusterBR> novModels = new ArrayList<>();
            
            //For each BR-model
            boolean removed = false;
            for (Map.Entry<String, ArrayList<MicroClusterBR>> listaMicroClusters : model.getModel().entrySet()) {
                //Silhouette validation
                if (currentEvaluatedMC.clusterSilhouette(listaMicroClusters.getValue()) > 0){
                    //********** The new micro-cluster is valid ****************
                    if(!removed){
                        removed = true;
                        toClassify = model.getInstancesUsedToBuildNewMC(removeExamples, indexMC);
                        currentEvaluatedMC.calculateAverOutputNovelty(toClassify);
                    }

                    //identifies the closer micro-cluster to the new valid micro-cluster 
//                        ret_func[0] <- posMinDist
//                        ret_func[1] <- minDist
//                        ret_func[0] <- threshold                        
                    double[] ret_func = identifyCloserMicroClusterTV1(currentEvaluatedMC, listaMicroClusters.getValue(), this.getThreashold());

                    //if dist < (stdDev * f) than extension, otherwise NP
                    if (ret_func[1] < ret_func[2]) {
                        textoArq = "Thinking " +
                                "extension: " + 
                                "C " +
                                listaMicroClusters.getKey() + 
                                " - " + 
                                (int) currentEvaluatedMC.getMicroCluster().getWeight() +
                                " examples";
                        extModels.add(listaMicroClusters.getValue().get((int)ret_func[0]));

                    } else {
                        textoArq = "Thinking " +
                                "np: " + 
                                "N " +
                                listaMicroClusters.getKey() + 
                                " - " + 
                                (int) currentEvaluatedMC.getMicroCluster().getWeight() + 
                                " examples";
                        novModels.add(listaMicroClusters.getValue().get((int)ret_func[0]));
                    }
                }
            } 

            Set<String> predictedLabels = new HashSet<>();
            if (!novModels.isEmpty() || !extModels.isEmpty()) {
                //If the number of models which considered the new micro-cluster as NP is greater than label cardinality
                if (extModels.size() >= model.getCurrentCardinality()) {
                    //Extension
                    this.getExtInfo().write("*********Extension**********"+"\n");
                    this.getExtInfo().write("Timestamp"+ this.getTimestamp() + "\n");
                    this.timeStampExtension.add(this.getTimestamp());
                    this.getTimeStampExtension().add(this.getTimestamp());
                    int i = 0;
                    while ( i < extModels.size()) {
                        //We must update mtxFrequencies and cardinality for calculating 
                        predictedLabels.add(extModels.get(i).getMicroCluster().getLabelClass());

                        for (Entry<Integer, Instance> inst: toClassify) {
//                            model.updateCurrentCardinality(predictedLabels.size());
//                            model.updateMtxFrequencies(predictedLabels);
                            Set<String> trueLabels = DataSetUtils.getLabelSet(inst.getValue())
                            model.addPrediction(trueLabels, predictedLabels);
                            model.incrementNumerOfObservedExamples();
                            model.removerUnknown(predictedLabels);
                            String textOut = "Ex: " + inst.getKey() + "\t True Labels: " + trueLabels.toString() + "\t MINAS-BR Prediction: ";
                            filePredictions.write(textOut + predictedLabels.toString() + "\n");
                            System.out.println(textOut + predictedLabels.toString());
                        }

                        if ((extModels.get(i).getMicroCluster().getCategory().equalsIgnoreCase("normal")) || 
                                (extModels.get(i).getMicroCluster().getCategory().equalsIgnoreCase("ext"))) {
                            //extension of a class learned offline
                            model.updateModel(currentEvaluatedMC, 
                                    extModels.get(i).getMicroCluster().getLabelClass(), 
                                    "ext", 
                                    timestamp);

                            textoArq = textoArq.concat("Thinking " + 
                                    "Extension: " +
                                    "C " + 
                                    extModels.get(i).getMicroCluster().getLabelClass() + 
                                    " - " + 
                                    (int) currentEvaluatedMC.getMicroCluster().getWeight() +
                                    " examples" + 
                                            "\n");

                        } else {
                            //extension of a novelty pattern
                            model.updateModel(currentEvaluatedMC, 
                                    extModels.get(i).getMicroCluster().getLabelClass(), 
                                    "extNov",
                                    this.timestamp);

                            textoArq = textoArq.concat("Thinking " + 
                                    "NoveltyExtension: " + 
                                    "N " + 
                                    extModels.get(i).getMicroCluster().getLabelClass() + 
                                    " - " +
                                    (int) currentEvaluatedMC.getMicroCluster().getWeight() +
                                    " examples");
                        }

                        i++;
                        this.getExtInfo().write(textoArq+"\n");
                    }
                } else {
                    //novelty pattern
                    this.getExtInfo().write("*********Novelty Pattern**********"+"\n");
                    this.getExtInfo().write("Timestamp"+ this.getTimestamp() + "\n");
                    this.timeStampNP.add(this.getTimestamp());

                    for (MicroClusterBR mc : extModels) {
                        predictedLabels.add(mc.getMicroCluster().getLabelClass());
                    }
                    predictedLabels.add("NP" + Integer.toString(model.getNPs().size() + 1));
                    
                    for (Entry<Integer, Instance> inst: toClassify) {
//                        model.updateCurrentCardinality(predictedLabels.size());
                        Set<String> trueLabels = DataSetUtils.getLabelSet(inst.getValue())
                        model.updateMtxFrequencies(predictedLabels);
                        model.addPrediction(trueLabels, predictedLabels);
                        model.incrementNumerOfObservedExamples();
                        model.removerUnknown(predictedLabels);
                        
                        String textOut = "Ex: " + inst.getKey() + "\t True Labels: " + trueLabels.toString() + "\t MINAS-BR Prediction: ";
                        filePredictions.write(textOut + predictedLabels.toString() + "\n");
                        System.out.println(textOut + predictedLabels.toString());
                    }
                    model.createModel(currentEvaluatedMC,extModels, this.timestamp, this.getExtInfo(), textoArq);
                }
                model.updateMicroClusterThresholds();
            }else
                System.out.println("None valid micro-clusters");
        }
        
        System.out.println(textoArq);
        this.fileOn.write(textoArq);
        this.fileOn.write("\n");
        //remove the examples marked with label -2, i. e., the unknown examples used in the creation of new valid micro-clusters
        int count_rem = 0;
        for (int g = 0; g < removeExamples.length; g++) {
            if (removeExamples[g] == -2) {
                model.getShortTimeMemory().getData().remove(g - count_rem);
                model.getShortTimeMemory().getTimestamp().remove(g - count_rem);
                count_rem++;
            }
        }
    }

    
    /**
     * Get micro-clusters of the models witch consider the NP as extension
     *
     * @param NP
     * @param modelo
     * @return
     */
    public ArrayList<Integer> getClosestsNPMicroClusters(MicroClusterBR NP, ArrayList<MicroClusterBR> modelo) {
        double distance = 0;
        double t = NP.getMicroCluster().getRadius() + Math.pow(NP.getMicroCluster().getRadius(), 2);
        ArrayList<Integer> closestMic = new ArrayList<Integer>();
        // calculates the distance between the center of the new cluster to the existing clusters 
        for (int i = 1; i < modelo.size(); i++) {
            distance = KMeansMOAModified.distance(modelo.get(i).getMicroCluster().getCenter(), NP.getMicroCluster().getCenter());
            if (distance < t) {
                closestMic.add(i);
            }
        }
        return closestMic;
    }


    /**
     * Identifica o micro-grupo mais próximo da novidade e calcula o limiar para
     * definir um NP
     *
     * @param modelUnk novidade
     * @param modelo modelo
     * @param threshold limiar estipulado pelo usuário
     * @return
     */
    public double[] identifyCloserMicroClusterTV1(MicroClusterBR modelUnk, ArrayList<MicroClusterBR> modelo, double threshold) {
        double[] ret = new double[3];
        double minDistance = Double.MAX_VALUE;
        int posMinDistance = 0;
        // calculate the distance between the center of the new cluster to the existing clusters 
        for (int i = 1; i < modelo.size(); i++) {
            double distance = KMeansMOAModified.distance(modelo.get(i).getMicroCluster().getCenter(), modelUnk.getMicroCluster().getCenter());
            if (distance < minDistance) {
                minDistance = distance;
                posMinDistance = i;
            }
        }
//        double vthreshold = (modelo.get(posMinDistance).getRadius()) * threshold;
        double vthreshold = modelo.get(posMinDistance).getMicroCluster().getRadius()/2 * threshold;
        if (minDistance < vthreshold) {
            ret[0] = posMinDistance;
            ret[1] = minDistance;
            ret[2] = vthreshold;
        } else { //intercept
            vthreshold = modelo.get(posMinDistance).getMicroCluster().getRadius() / 2 + modelUnk.getMicroCluster().getRadius() / 2;
            ret[0] = posMinDistance;
            ret[1] = minDistance;
            ret[2] = vthreshold;
        }
        return ret;
    }

    /**
     * Identifies if the new micro-clusters are NP or Extension using the TV2
     * strategy (max distance between the closest micro-clusters and the others)
     *
     * @param modelUnk novidade
     * @param modelo modelo
     * @param threshold limiar estipulado pelo usuário
     * @return
     */
    public Voting identifyCloserMicroClusterTV2(MicroClusterBR modelUnk, ArrayList<MicroClusterBR> modelo, double threshold) {
        double minDistance = Double.MAX_VALUE;
        int posMinDistance = 0;
        // calculates the distance between the center of the new cluster to the existing clusters 
        for (int i = 1; i < modelo.size(); i++) {
            double distance = KMeansMOAModified.distance(modelo.get(i).getMicroCluster().getCenter(), modelUnk.getMicroCluster().getCenter());
            if (distance < minDistance) {
                minDistance = distance;
                posMinDistance = i;
            }
        }
        double vthreshold = minDistance;
        for (int i = 1; i < modelo.size(); i++) {
            double distance = KMeansMOAModified.distance(modelo.get(i).getMicroCluster().getCenter(), modelo.get(posMinDistance).getMicroCluster().getCenter());
            if (distance > vthreshold) {
                vthreshold = distance;
            }
        }
        String categoria = modelo.get(posMinDistance).getMicroCluster().getCategory();
        String classe = modelo.get(posMinDistance).getMicroCluster().getLabelClass();

        return new Voting(categoria, classe, minDistance, posMinDistance, vthreshold);
    }
    /**
     * Identifies if the new micro-clusters are NP or Extension using the TV3
     * strategy (mean between the Euclidian distance of the closest micro-clusters and the others)
     *
     * @param modelUnk novidade
     * @param modelo modelo
     * @param threshold limiar estipulado pelo usuário
     * @return
     */
    public Voting identifyCloserMicroClusterTV3(MicroClusterBR modelUnk, ArrayList<MicroClusterBR> modelo, double threshold) {
        double minDistance = Double.MAX_VALUE;
        int posMinDistance = 0;
        double distance = 0;
        double sumDistance = 0;
        for (int i = 1; i < modelo.size(); i++) {
            sumDistance += KMeansMOAModified.distance(modelo.get(i).getMicroCluster().getCenter(), modelUnk.getMicroCluster().getCenter());
        }
        double vthreshold = sumDistance / modelo.size();
        for (int i = 1; i < modelo.size(); i++) {
            distance = KMeansMOAModified.distance(modelo.get(i).getMicroCluster().getCenter(), modelo.get(posMinDistance).getMicroCluster().getCenter());
            if (distance < minDistance) {
                minDistance = distance;
                posMinDistance = i;
            }
        }
        String categoria = modelo.get(posMinDistance).getMicroCluster().getCategory();
        String classe = modelo.get(posMinDistance).getMicroCluster().getLabelClass();

        return new Voting(categoria, classe, minDistance, posMinDistance, vthreshold);
    }

    
    
    /**
     * Identifies which models explain an example
     *
     * @param data
     * @param model
     * @return 
     */
    public ArrayList<Voting> identifyExample(Instance data, Model model) {
        ArrayList<Voting> voting = new ArrayList<>();
        for (Map.Entry<String, ArrayList<MicroClusterBR>> listaMicroClusters : model.getModel().entrySet()) {
            double distance = 0.0;
            assert(listaMicroClusters.getValue().size() > 0);
            
            String key = listaMicroClusters.getKey();
            double minDist = Double.MAX_VALUE;
            int posMinDist = 0;
            int pos = 0;
            for (MicroClusterBR microCluster : listaMicroClusters.getValue()) {
                double[] aux = Arrays.copyOfRange(data.toDoubleArray(), data.numOutputAttributes(), data.numAttributes());
                distance = KMeansMOAModified.distance(aux, microCluster.getMicroCluster().getCenter());
                if(distance < minDist){
                    minDist = distance;
                    posMinDist = pos;
                }
                pos++;
            }

            if (minDist <= listaMicroClusters.getValue().get(posMinDist).getMicroCluster().getRadius()) {
                Voting result = new Voting();
                result.setlabel(key);
                result.setCategory(listaMicroClusters.getValue().get(posMinDist).getMicroCluster().getCategory()); //Normal, extension ou novelty
                result.setDistance(minDist);

                /*add the examples in micro-clusters*/
                //                        double[] aux = Arrays.copyOfRange(data.toDoubleArray(), this.qtdeTotalClasses, data.numAttributes());
                //                        Instance inst = new DenseInstance(1, aux);
                //                        listaMicroClusters.getValue().get(posMinDistance).insert(inst, this.timestamp);
                //                    System.out.println("Classificado como: " + key + "\n");

                voting.add(result);

                listaMicroClusters.getValue().get(posMinDist).getMicroCluster().setTime(this.getTimestamp());
            }
        }
        return voting;
    }

    /**
     * @return the threshold
     */
    public double getThreashold() {
        return getThreshold();
    }

    /**
     * @return the qtdeExExcluidos
     */
    public int getQtdeExExcluidos() {
        return qtdeExExcluidos;
    }

    /**
     * Gets the biggest micro-clusters diameter
     *
     * @param keySet
     * @return
     */
    private double getMaxDiameter(Set<MicroClusterBR> keySet) {
        double maxD = 0;
        for (Iterator<MicroClusterBR> iterator = keySet.iterator(); iterator.hasNext();) {
            MicroClusterBR next = iterator.next();
            maxD = next.getMicroCluster().getRadius() * 2;
            if (maxD < (next.getMicroCluster().getRadius() * 2)) {
                
            }
        }
        return maxD;
    }
    
    public void removeOldMicroClusters(int windowSize, Model modelo, FileWriter fileOut) throws IOException {
        ArrayList<MicroClusterBR> listaMicro = new ArrayList<MicroClusterBR>();
        ArrayList<String> classToRemove = new ArrayList<>();
//        try{
            for (Map.Entry<String, ArrayList<MicroClusterBR>> entry : modelo.getModel().entrySet()) {
                String key = entry.getKey();
                ArrayList<MicroClusterBR> value = entry.getValue();
                for (int i = 0; i < value.size(); i++) {
                    if (value.get(i).getMicroCluster().getTime() <= (this.getTimestamp() - (windowSize))) {
                        listaMicro.add(value.get(i));
                        this.getSleepMemory().add(value.get(i));
                        fileOut.write("Removed micro-clusters: " + i +" label: " + value.get(i).getMicroCluster().getLabelClass() + " category: " +  value.get(i).getMicroCluster().getCategory());
                        fileOut.write("\n");
                        value.remove(i);
                        i--;
                    }
                }
                try{
                     fileOut.write("Timestamp: " + this.getTimestamp() + " - removed micro-clusters: " + listaMicro.size() + " - model's size ["+value.get(0).getMicroCluster().getLabelClass()+"]:" + value.size() + "\n");
                    System.out.println("Timestamp: " + this.getTimestamp() + " - removed micro-clusters: " + listaMicro.size() + " - model's size ["+value.get(0).getMicroCluster().getLabelClass()+"]:" + value.size());
                }catch(IndexOutOfBoundsException e){
                    
                    classToRemove.add(key);
                }
            }
            for (String classe : classToRemove) {
                System.err.println("[WARNING]: All micro-clusters assigned with label " + classe+ " gonna be removed");
                modelo.getModel().remove(classe);
            }
    }
    

    /**
     * @return the exShortTimeMem
     */
    public int getExShortTimeMem() {
        return exShortTimeMem;
    }


    /**
     * @return the threshold
     */
    public double getThreshold() {
        return threshold;
    }

    /**
     * @return the lastCheck
     */
    public int getLastCheck() {
        return lastCheck;
    }

    /**
     * @param lastCheck the lastCheck to set
     */
    public void setLastCheck(int lastCheck) {
        this.lastCheck = lastCheck;
    }


    /**
     * @return the timeStampExtension
     */
    public ArrayList<Integer> getTimeStampExtension() {
        return timeStampExtension;
    }

    /**
     * @param timeStampExtension the timeStampExtension to set
     */
    public void setTimeStampExtension(ArrayList<Integer> timeStampExtension) {
        this.timeStampExtension = timeStampExtension;
    }


    /**
     * @return the timeStampNP
     */
    public ArrayList<Integer> getTimeStampNP() {
        return timeStampNP;
    }

    /**
     * @return the extInfo
     */
    public FileWriter getExtInfo() {
        return extInfo;
    }
    
    private HashMap<Integer,MicroClusterBR> createModelKMeansLeader(Model model, int[] exampleCluster, int timestamp) throws NumberFormatException, IOException {
        List<ClustreamKernelMOAModified> examples = new LinkedList<>();

        //Adicionando os exemplos ao algoritmo
        for (int k = 0; k < model.getShortTimeMemory().getData().size(); k++) {
            double[] data = Arrays.copyOfRange(model.getShortTimeMemory().getData().get(k).toDoubleArray(),
                    model.getShortTimeMemory().getData().get(k).numOutputAttributes(),
                    model.getShortTimeMemory().getData().get(k).numAttributes());
            Instance inst = new DenseInstance(1, data);
            examples.add(new ClustreamKernelMOAModified(inst, inst.numAttributes(), k));
        }

        //********* K-Means ***********************
        //generate initial centers with leader algorithmn
        ArrayList<Integer> centroids = this.leaderAlgorithm(model.getShortTimeMemory().getData(), model.getGlobalMaxRadius());
        ClustreamKernelMOAModified[] centrosIni = new ClustreamKernelMOAModified[centroids.size()];
        for (int i = 0; i < centroids.size(); i++) {
            centrosIni[i] = examples.get(centroids.get(i));
        }

        Clustering centers;
        moa.clusterers.KMeans cm = new moa.clusterers.KMeans();
        centers = cm.kMeans(centrosIni, examples);

        //*********results     
        // transform the results of kmeans in a data structure used by MINAS
        CFCluster[] res = new CFCluster[centers.size()];
        for (int j = 0; j < examples.size(); j++) {
            // Find closest kMeans cluster
            double minDistance = Double.MAX_VALUE;
            int closestCluster = 0;
            for (int i = 0; i < centers.size(); i++) {
                double distance = KMeansMOAModified.distance(centers.get(i).getCenter(), examples.get(j).getCenter());
                if (distance < minDistance) {
                    closestCluster = i;
                    minDistance = distance;
                }
            }

            // add to the cluster
            if (res[closestCluster] == null) {
                res[closestCluster] = (CFCluster) examples.get(j).copy();
            } else {
                res[closestCluster].add(examples.get(j));
            }
            exampleCluster[j] = closestCluster;
        }
        
        Clustering micros = new Clustering(res);

        HashMap<Integer, MicroClusterBR> modelSet = new HashMap<>();
        for (int w = 0; w < micros.size(); w++) {
            if(micros.get(w) != null && ((ClustreamKernelMOAModified) micros.get(w)).getN() > 3){
                MicroClusterBR model_tmp = new MicroClusterBR(new MicroCluster((ClustreamKernelMOAModified) micros.get(w), "", "", timestamp));
                modelSet.put(w,model_tmp);
            }
        }
        return modelSet;
    }
    
    private ArrayList<Integer> leaderAlgorithm(ArrayList<Instance> dataSet, double maxRadius) {
        ArrayList<Integer> centroids = new ArrayList<Integer>();
        Random random = new Random();
        random.setSeed(420);
        centroids.add(random.nextInt(dataSet.size()));
        
        for (int i = 0; i < dataSet.size(); i++) {
            boolean centroid = false;
            for (int j = 0; j < centroids.size(); j++) {
                double[] data1 = Arrays.copyOfRange(dataSet.get(i).toDoubleArray(),
                        dataSet.get(i).numOutputAttributes(), 
                        dataSet.get(i).numAttributes()
                );
                double[] data2 = Arrays.copyOfRange(dataSet.get(centroids.get(j)).toDoubleArray(), 
                        dataSet.get(i).numOutputAttributes(), 
                        dataSet.get(centroids.get(j)).numAttributes()
                );
                double dist = KMeansMOAModified.distance(data1, data2);
                if(dist < maxRadius){
                    centroid = false;
                    break;
                }else{
                    centroid = true;
                }
            }
            if(centroid){
                centroids.add(i);
            }
        }
        return centroids;
    }

    
}
