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
import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.clusterers.KMeans;

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
        
        List<ClustreamKernelMOAModified> examples = new LinkedList<>();
        if(numMClusters < 1){
            numMClusters = 1;
        }
        int indexLabels = dataSet.get(0).numOutputAttributes();
        int numAtt =  dataSet.get(0).numAttributes() - indexLabels;
        
        
        //************read dataset *************************
        //read examples from the file to the memory to execute Kmeans
        
        dataSet.stream().forEach(inst -> {
            double[] data = Arrays.copyOfRange(inst.toDoubleArray(), inst.numOutputAttributes(), inst.numAttributes());
            Instance inst2 = new DenseInstance(1, data);
            examples.add(new ClustreamKernelMOAModified(inst2, numAtt, 0));
        });

        //********* K-Means ***********************
        //generate initial centers aleatory
        ClustreamKernelMOAModified[] centrosIni = new ClustreamKernelMOAModified[numMClusters];
        List<Integer> numeros = new ArrayList<Integer>();
        int cont = 0;
        for (int c = 0; c < examples.size(); c++) {
            numeros.add(c);
        }
        Collections.shuffle(numeros);
        
        for (int i = 0; i < numMClusters; i++) {
            centrosIni[i] = examples.get(numeros.get(i));
        }

        //execution of the KMeans  
        moa.clusterers.KMeans cm = new moa.clusterers.KMeans();
       Clustering centers = cm.kMeans(centrosIni,examples);  

        //*********results     
        // transform the results of kmeans in a data structure used by MINAS
        HashMap<Integer, ArrayList<double[]>> mcInstances = new HashMap<>();
        CFCluster[] res = new CFCluster[centers.size()];
        examples.stream().forEach(example -> {
//        for (int j = 0; j < examples.size(); j++) {
            // Find closest kMeans cluster
            double minDistance = Double.MAX_VALUE;
            int closestCluster = 0;
            for (int i = 0; i < centers.size(); i++) {
                double distance = KMeansMOAModified.distance(centers.get(i).getCenter(), example.getCenter());
                if (distance < minDistance) {
                    closestCluster = i;
                    minDistance = distance;
                }
            }
            
           // add to the cluster
            if (res[closestCluster] == null) {
                res[closestCluster] = (CFCluster) example.copy();
            } else {
                res[closestCluster].add(example);
            }

            try{
                mcInstances.get(closestCluster).add(example.getCenter());
            }catch(NullPointerException e){
                mcInstances.put(closestCluster, new ArrayList<>());
                mcInstances.get(closestCluster).add(example.getCenter());
            }
        });
        
        Clustering micros = new Clustering(res);
        ArrayList<MicroClusterBR> modelSet = new ArrayList<>();
        for (int w = 0; w < micros.size(); w++) {
            //Do not considering non-representative clusters
            if(micros.get(w) != null && ((ClustreamKernelMOAModified) micros.get(w)).getN() > 3){
                MicroClusterBR model_tmp = new MicroClusterBR(new MicroCluster((ClustreamKernelMOAModified)micros.get(w),label,"normal",0));   
                model_tmp.calculateInitialAverOutput(mcInstances.get(w));
                model_tmp.calculateThreshold(model.getMtxLabelsFrequencies(), model.getNumberOfObservedExamples());
                modelSet.add(model_tmp);
            }
        }
        return modelSet;
    }
    

    /**
     * @return the trainingData
     */
    public HashMap<String, ArrayList<Instance>> getTrainingData() {
        return trainingData;
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
            int[] clusteringResult = new int[subconjunto.size()];
            
            ArrayList<MicroClusterBR> clusterSet = this.createModelKMeansOffline(subconjunto, 
                    key, 
                    clusteringResult,
                    (int) Math.ceil(subconjunto.size() * k_ini));
            
            model.getModel().put(key, clusterSet);
            System.out.println("Class: " + key + " size: " + clusterSet.size() + " n:" + subconjunto.size());
            this.fileOut.write("Class: " + key + " size: " + clusterSet.size() + " n:" + subconjunto.size() + "\n");
        }
        model.setClasses(this.getTrainingData().keySet());
        model.writeBayesRulesElements(0, this.directory);
    }
    
    /**
     * Separa os exemplos em subconjuntos, um para cada classe
     * @param D conjunto de treino
     * @throws Exception 
     */
    public void setTrainingData(ArrayList<Instance> D) throws Exception{
        model = new Model();
        model.inicialize(this.directory);
        int qtdeRotulos = 0;
        HashMap<String, ArrayList<Instance>> trainingData = new HashMap<>();

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
        this.model.setCurrentCardinality(Math.ceil((double)qtdeRotulos/(double)D.size()));
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
