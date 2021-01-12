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
//import evaluate.FreeChartGraph;
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
import java.util.Set;
import moa.cluster.CFCluster;
import moa.cluster.Clustering;
import utils.OnlinePhaseUtils;

/**
 * 
 * @author joel
 * 
 */
public final class OfflinePhase{
    private Model model;
    private double k_ini;
     private HashMap<String, ArrayList<Instance>> trainingData;
    private String algOff;                                      //algoritmo de agrupamento na fase offline  
    private FileWriter fileOut;
    private String directory;
    
    public OfflinePhase(ArrayList<Instance> trainingFile,
            double k_ini, 
            FileWriter fileOff, 
            String outputDirectory) throws Exception{
        
        this.algOff = "kmeans";
        this.fileOut = fileOff;
        this.directory = outputDirectory;
        this.setK_ini(k_ini);
        this.setTrainingData(trainingFile);
        this.training();
        fileOff.write("Label Cardinality: " + this.model.getCurrentCardinality() + "\n");
    }
    
    public ArrayList<MicroClusterBR> createModelKMeansOffline(ArrayList<Instance> dataSet, String label, int[] exampleCluster, int numMClusters) throws NumberFormatException, IOException {
        ArrayList<MicroClusterBR> modelSet = new ArrayList<MicroClusterBR>();
        List<ClustreamKernelMOAModified> examples = new LinkedList<ClustreamKernelMOAModified>();
        if(numMClusters < 1){
            numMClusters = 1;
        }
        int indexLabels = dataSet.get(0).numOutputAttributes();
        int numAtt =  dataSet.get(0).numAttributes() - indexLabels;
        
        
        //************read dataset *************************
        //read examples from the file to the memory to execute Kmeans
        for (int k = 0; k < dataSet.size(); k++) {
//            System.out.println(Arrays.toString(dataSet.get(k).toDoubleArray()));
            double[] data = Arrays.copyOfRange(dataSet.get(k).toDoubleArray(), dataSet.get(k).numOutputAttributes(), dataSet.get(k).numAttributes());
            Instance inst = new DenseInstance(1, data);
            examples.add(new ClustreamKernelMOAModified(inst, numAtt, 0));
            exampleCluster = new int[dataSet.size()];
        }

        //********* K-Means ***********************
        //generate initial centers aleatory
        ClustreamKernelMOAModified[] centrosIni = new ClustreamKernelMOAModified[numMClusters];
        int nroaleatorio;
        List<Integer> numeros = new ArrayList<Integer>();
        for (int c = 0; c < examples.size(); c++) {
            numeros.add(c);
        }
        Collections.shuffle(numeros);
        for (int i = 0; i < numMClusters; i++) {
            nroaleatorio = numeros.get(i).intValue();
            centrosIni[i] = examples.get(i/* nroaleatorio*/);
        }

        //execution of the KMeans  
        Clustering centers;
        moa.clusterers.KMeans cm = new moa.clusterers.KMeans();
//        try{
            centers = cm.kMeans(centrosIni, examples);
//        }catch(Exception e){
//            System.out.println("");
//        }
        
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
        ArrayList<ArrayList<Integer>> mapClustersExamples = new ArrayList<ArrayList<Integer>>();
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
                model_tmp = new MicroClusterBR(new MicroCluster((ClustreamKernelMOAModified) micros.get(w), label, "normal", 0));
                modelSet.add(model_tmp);
            }
        }
        return modelSet;
    }
    
    /**
     * Cria modelo de decisão através do algoritmo CluStream
     * @param examples of class
     * @param label of class
     * @param numMClusters
     * @return a list of microclusters that represents a class
     * @throws NumberFormatException
     * @throws IOException 
     */
    public ArrayList<MicroClusterBR> criarmodeloCluStreamOffline(ArrayList<Instance> examples, String label, int numMClusters) throws NumberFormatException, IOException {
        ArrayList<MicroClusterBR> conjModelos = new ArrayList<>();
        ClustreamOfflineBR jc = new ClustreamOfflineBR();

        Clustering micros = jc.CluStream(examples, examples.get(0).numOutputAttributes(), numMClusters, true, true /*executa kmeans*/);

        for (int w = 0; w < micros.size(); w++) {
            MicroClusterBR mdtemp = new MicroClusterBR(new MicroCluster((ClustreamKernelMOAModified) micros.get(w), label, "normal", 0));
            // add the temporary model to the decision model 
            conjModelos.add(mdtemp);
        }
        return conjModelos;
    }

    /**
     * @return the trainingData
     */
    public HashMap<String, ArrayList<Instance>> getTrainingData() {
        return trainingData;
    }

    /**
     * @param trainingData the trainingData to set
     */
    public void setTrainingData(HashMap<String, ArrayList<Instance>> trainingData) {
        this.trainingData = trainingData;
    }

    /**
     * @return the algOff
     */
    public String getAlgOff() {
        return algOff;
    }

    /**
     * @return the fileOut
     */
    public FileWriter getFileOut() {
        return fileOut;
    }

    /**
     * @return the directory
     */
    public String getDirectory() {
        return directory;
    }

    /**
     * @param directory the directory to set
     */
    public void setDirectory(String directory) {
        this.directory = directory;
    }
    
    
    
    /**
     * Builds the model
     * @throws IOException
     * @throws Exception 
     */
    public void training() throws IOException, Exception {
        //create one training file for each problem class 
        System.out.print("Classes for the training phase (offline): ");
        this.fileOut.write("Classes for the training phase (offline): ");
        System.out.println("" + this.trainingData.keySet().toString());
        this.fileOut.write("" + this.trainingData.keySet().toString());
        System.out.print("\nQuantidade de classes: ");
        this.fileOut.write("\nQuantidade de classes: ");
        System.out.println("" + this.trainingData.size());
        this.fileOut.write("" + this.trainingData.size() + "\n");
        
        
        //generate a set of micro-clusters for each class from the training set
        for(Map.Entry<String, ArrayList<Instance>> entry : this.trainingData.entrySet()) {
            String key = entry.getKey();
            ArrayList<Instance> subconjunto = entry.getValue();
            ArrayList<MicroClusterBR> clusterSet = null;
            int[] clusteringResult = new int[subconjunto.size()];
            
            clusterSet = this.createModelKMeansOffline(subconjunto, 
                    key, 
                    clusteringResult,
                    (int) Math.ceil(subconjunto.size() * k_ini));
            
            model.getModel().put(key, clusterSet);
            System.out.println("Class: " + key + " size: " + clusterSet.size() + " n:" + subconjunto.size());
            this.fileOut.write("Class: " + key + " size: " + clusterSet.size() + " n:" + subconjunto.size() + "\n");
        }
        model.setClasses(this.getTrainingData().keySet());
    }
    
    /**
     * Separa os exemplos em subconjuntos, um para cada classe
     * @param D conjunto de treino
     * @param classesConhecidas
     * @throws Exception 
     */
    public void setTrainingData(ArrayList<Instance> D) throws Exception{
        model = new Model();
        model.inicialize(this.directory);
        int qtdeRotulos = 0;
        HashMap<String, ArrayList<Instance>> trainingData = new HashMap<String, ArrayList<Instance>>();

        for (int i = 0; i < D.size(); i++) {
            Set<String> labels = DataSetUtils.getLabelSet(D.get(i)); 
            qtdeRotulos += labels.size();
            ArrayList<Instance> generic; 
            
            //For each label assigned to an example, add this example into it repesctive set.
            for (String label : labels) { 
                ArrayList<Instance> dataset;
                try{
                    dataset = trainingData.get(label); 
                    dataset.add(D.get(i));
                }catch(NullPointerException e){
                    System.out.println("Create new set for label: " + label);
                    dataset = new ArrayList<>();
                    dataset.add(D.get(i));
                }
                trainingData.put(label,dataset);
                
                //Filling the matrix T (frequencies)
                for (String label_column : labels) { 
                    String mtxCordinate = label+","+label_column;
                    int frequency = 0;
                    try{
                        frequency = model.getMtxLabelsFrequencies().get(mtxCordinate);
                        frequency ++;
                        model.getMtxLabelsFrequencies().put(mtxCordinate, frequency);
                    }catch(NullPointerException e){
                        model.getMtxLabelsFrequencies().put(mtxCordinate, 1);
                    }
                }
            }
            this.model.incrementNumerOfObservedExamples();
        }
        this.trainingData = trainingData;
        this.model.setInitialProbabilities();
        this.model.setCurrentCardinality(Math.ceil(qtdeRotulos/D.size()));
    }
    

    /**
     * @return the model
     */
    public Model getModel() {
        return model;
    }

    /**
     * @param k_ini the k_ini to set
     */
    public void setK_ini(double k_ini) {
        this.k_ini = k_ini;
    }

}
