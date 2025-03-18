/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package br;

import NoveltyDetection.KMeansMOAModified;
import NoveltyDetection.MicroCluster;
import com.yahoo.labs.samoa.instances.Instance;
import dataSource.DataSetUtils;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import utils.ShortTimeMemory;
import utils.Voting;

/**
 *
 * @author joel
 */
public class Model {
    private String outputDirectory;
    private int evaluationWindowSize;
    private HashMap<String, ArrayList<MicroClusterBR>> model;
    private ArrayList<NoveltyPattern> NPs;
    private ArrayList<Class> Classes;
    private ArrayList<Integer> timestampNP;
    private int evaluatedExamples;
    private ArrayList<Set<String>> Yall; //All true labels in the window
    private ArrayList<Set<String>> Zall; // All prediction in the window
    private ArrayList<Set<String>> classifiedUnk = new ArrayList<>(); //Unknown examples classified during the ND procedure
    private FileWriter pnInfo;
    private ShortTimeMemory shortTimeMemory = new ShortTimeMemory(new ArrayList<>(), new ArrayList<>());
    private HashMap<String, Integer> mtxLabelsFrequencies;
    private HashMap<String, Double> mtxProbabilities;
    private double currentCardinality;
    private int numberOfObservedExamples;

    public int getNumberOfObservedExamples() {
        return numberOfObservedExamples;
    }

    public double getCurrentCardinality() {
        return currentCardinality;
    }

    public void setCurrentCardinality(double currentCardinality) {
        this.currentCardinality = currentCardinality;
    }
    
    public HashMap<String, Integer> getMtxLabelsFrequencies() {
        return mtxLabelsFrequencies;
    }

    public HashMap<String, Double> getMtxProbabilities() {
        return mtxProbabilities;
    }
    
    public void incrementNumerOfObservedExamples() {
        this.numberOfObservedExamples++;
    }
    
    public int getEvaluationWindowSize() {
        return evaluationWindowSize;
    }

    public void setEvaluationWindowSize(int evaluationWindowSize) {
        this.evaluationWindowSize = evaluationWindowSize;
    }

    public void updateMtxFrequencies(Set<String> Z) {
        for (String j : Z) {
            for (String n : Z) {
                try{
                    int frequency = this.mtxLabelsFrequencies.get(j+","+n) + 1;
                    this.mtxLabelsFrequencies.put(j+","+n, frequency);
                }catch(NullPointerException e){
                    this.mtxLabelsFrequencies.put(j+","+n, 1);
                }
            }
        }
        
//        Z.parallelStream().forEach(j -> {
//            Z.parallelStream().forEach(n -> {
//                try{
//                    int frequency = this.mtxLabelsFrequencies.get(j+","+n) + 1;
//                    this.mtxLabelsFrequencies.put(j+","+n, frequency);
//                }catch(NullPointerException e){
//                    this.mtxLabelsFrequencies.put(j+","+n, 1);
//                }
//            });
//        });
    }

    public void writeBayesRulesElements(int timestamp, String outputDirectory) {
        try {
            FileWriter file = null;
            if(timestamp <= 0){
                file = new FileWriter(new File(outputDirectory+"/thresholdsInfo.csv"), false);
                file.write("timestamp,threshold,averOut,label" +"\n");
            }else{
               file = new FileWriter(new File(outputDirectory+"/thresholdsInfo.csv"), true);
            }
            
            
            for (Map.Entry<String, ArrayList<MicroClusterBR>> entry : model.entrySet()) {
                ArrayList<MicroClusterBR> mcSet = entry.getValue();
                for (MicroClusterBR mc : mcSet) {
                    file.write(timestamp+","+mc.getThreshold()+","+mc.getAverOut()+","+mc.getMicroCluster().getLabelClass()+"\n");
                }
            }
            file.close();
        } catch (IOException ex) {
            Logger.getLogger(Model.class.getName()).log(Level.SEVERE, null, ex);
            System.err.println("Falha no arquivo thresholdsInfo.csv");
        }
    }
    
    public void writeCurrentCardinality(int timestamp, String outputDirectory) throws IOException{
        FileWriter file = null;
        try {
            if(timestamp <= 1){
                file = new FileWriter(new File(outputDirectory + "/cardinalitiesOverTime.csv"), false);
                file.write("timestamp,cardinality" +"\n");
            }else{
                file = new FileWriter(new File(outputDirectory + "/cardinalitiesOverTime.csv"), true);
            }
                
        } catch (IOException ex) {
            Logger.getLogger(Model.class.getName()).log(Level.SEVERE, null, ex);
            System.err.println("Falha no arquivo cardinalitiesOverTime.csv");
        }
        
        
        file.write(timestamp + "," + this.currentCardinality + "\n");
        file.close();
    }
    
    public ArrayList<Voting> getClosestMicroClusters(Instance data, int k) {
        ArrayList<Voting> voting = new ArrayList<>();
        for (Map.Entry<String, ArrayList<MicroClusterBR>> microClustersList : this.getModel().entrySet()) {
            int cont = 1;
            ArrayList<MicroClusterBR>  removableMicroClusterList = null;
            try{
                removableMicroClusterList =  new ArrayList(microClustersList.getValue());
            }catch(NullPointerException e){
                System.out.println("ModelSet empty --> getClosestMicroClusters function");
                e.printStackTrace();
                System.exit(0);
            }
            
            //Getting the k-nearst micro-clusters and put them into Voting
            while(cont <= k && !removableMicroClusterList.isEmpty()){
                double distance = 0.0;
                String key = microClustersList.getKey();
                double minDist = Double.MAX_VALUE;
                int posMinDist = 0;
                for (int i = 0; i < removableMicroClusterList.size(); i++) {
                    double[] aux = Arrays.copyOfRange(data.toDoubleArray(), data.numOutputAttributes(), data.numAttributes());
                    distance = KMeansMOAModified.distance(aux, removableMicroClusterList.get(i).getMicroCluster().getCenter());
                    if(distance < minDist){
                        minDist = distance;
                        posMinDist = i;
                    }
                }

                if (minDist <= removableMicroClusterList.get(posMinDist).getMicroCluster().getRadius()) {
                    Voting result = new Voting();
                    result.setlabel(key);
                    result.setCategory(removableMicroClusterList.get(posMinDist).getMicroCluster().getCategory()); //Normal, extension ou novelty
                    result.setDistance(minDist);
                    result.setPosMC(posMinDist);

                    /*add the examples in micro-clusters*/
                    //                        double[] aux = Arrays.copyOfRange(data.toDoubleArray(), this.qtdeTotalClasses, data.numAttributes());
                    //                        Instance inst = new DenseInstance(1, aux);
                    //                        listaMicroClusters.getValue().get(posMinDistance).insert(inst, this.timestamp);
                    //                    System.out.println("Classificado como: " + key + "\n");

                    voting.add(result);
//                    microClustersList.getValue().get(posMinDist).getMicroCluster().setTime(timestamp);
                    removableMicroClusterList.remove(posMinDist);
                }
                cont++;
            }
        }
        return voting;
    }

    public Set<String> bayesRuleToClassify(ArrayList<Voting> voting, Instance x_i, int timestamp) {
        Collections.sort(voting);
        Set<String> Y_pred = new HashSet<>();
        double[] x_i_inputs = Arrays.copyOfRange(x_i.toDoubleArray(), x_i.numOutputAttributes(), x_i.numAttributes());
        
        Y_pred.add(this.getModel().get(voting.get(0).getlabel()).
                get(voting.get(0).getPosMC()).
                getMicroCluster().
                getLabelClass()
        );
        
        double p_xi_yc = Math.exp(-KMeansMOAModified.distance(x_i_inputs, 
                this.getModel().get(voting.get(0).getlabel()).
                        get(voting.get(0).getPosMC()).
                        getMicroCluster().getCenter()
                ));
        
        this.getModel().get(voting.get(0).getlabel()).
                get(voting.get(0).getPosMC()).
                getMicroCluster().setTime(timestamp);
        
        this.getModel().get(voting.get(0).getlabel()).
                get(voting.get(0).getPosMC()).updateAverOut(p_xi_yc);
        
        this.getModel().get(voting.get(0).getlabel()).
                get(voting.get(0).getPosMC()).
                calculateThreshold(mtxLabelsFrequencies, this.numberOfObservedExamples);
        
        for (int i = 1; i < voting.size(); i++) {
            if(Y_pred.contains(voting.get(i).getlabel()))
                continue;
            double p_yc = this.getPriorProbability(voting.get(i).getlabel());
            
            
            MicroClusterBR winMC = this.getModel().get(voting.get(i).getlabel()).
                    get(voting.get(i).getPosMC());
            p_xi_yc = Math.exp(-KMeansMOAModified.distance(x_i_inputs, winMC.getMicroCluster().getCenter()));
            
            double prod = 1;
            for (String y_k : Y_pred) {
                double p_yk_yc = this.getPosteriorProbability(y_k, voting.get(i).getlabel());
                prod *= p_yk_yc;
            }
            
            double proba = p_yc * prod * p_xi_yc;
            
            if(proba >= winMC.getThreshold()){
                Y_pred.add(winMC.getMicroCluster().getLabelClass());
//                winMC.getMicroCluster().insert(x_i_inputs, timestamp);
                winMC.updateAverOut(Math.exp(-KMeansMOAModified.distance(x_i_inputs, winMC.getMicroCluster().getCenter())));
                winMC.calculateThreshold(mtxLabelsFrequencies, this.numberOfObservedExamples);
                winMC.getMicroCluster().setTime(timestamp);
            }
        }
        
        return Y_pred;
    }
    
    

    private double getPriorProbability(String c) {
        return (double)this.mtxLabelsFrequencies.get(c+","+c) / (double)this.numberOfObservedExamples;
    }

    private double getPosteriorProbability(String y_k, String y_c) {
        try{
            return (double) this.mtxLabelsFrequencies.get(y_k+","+y_c) / (double) this.mtxLabelsFrequencies.get(y_c+","+y_c);
        }catch(NullPointerException e){
            return 1;
        }
    }

//    public void updateCurrentCardinality(int z_new) {
//        this.currentCardinality =  ((double) this.numberOfObservedExamples * this.currentCardinality + (double) z_new) /
//                ((double)this.numberOfObservedExamples + (double)z_new);
//    }
    
    /**
     * Get the greatest micro-cluster radius of the model
     * @param modelo
     * @return 
     */
    public double getGlobalMaxRadius() {
        double maxRadius = 0;
        for (Map.Entry<String, ArrayList<MicroClusterBR>> entry : this.model.entrySet()) {
            ArrayList<MicroClusterBR> value = entry.getValue();
            if(maxRadius < this.getLocalMaxRadius(value)){
                maxRadius = this.getLocalMaxRadius(value);
            }
        }
        return maxRadius;
    }
    
    
    /**
     * Get the greatest micro-cluster radius of the model
     * @param modelo
     * @return 
     */
    private double getLocalMaxRadius(ArrayList<MicroClusterBR> modelo) {
        double maxRadius = 0;
        for (MicroClusterBR m : modelo) {
            if(m.getMicroCluster().getRadius() > maxRadius)
                maxRadius = m.getMicroCluster().getRadius();
        }
        return maxRadius;
    }
    
    /**
     * creates a new model to represent the new Novelty Pattern
     * @param model
     * @param novelty
     * @param noveltyPatternsControl
     * @param textoArq
     * @throws java.io.IOException
     */
    public void createModel(MicroClusterBR currentEvaluatedMC, ArrayList<MicroClusterBR> extension, int timestamp, FileWriter fileOut, String textoArq) throws NumberFormatException, IOException {
        
        currentEvaluatedMC.getMicroCluster().setTime(timestamp);
        currentEvaluatedMC.getMicroCluster().setCategory("nov");
        currentEvaluatedMC.getMicroCluster().setLabelClass("NP" + Integer.toString(this.getNPs().size() + 1));
//        currentEvaluatedMC.calculateThreshold(mtxLabelsFrequencies, numberOfObservedExamples);
        
        ArrayList<MicroClusterBR> novelties = new ArrayList<>();
        novelties.add(currentEvaluatedMC);
        
        for (MicroClusterBR mc : extension) {
            MicroClusterBR np = new MicroClusterBR((MicroCluster) mc.getMicroCluster().copy());
            np.setThreshold(mc.getThreshold());
            np.setAverOut(np.getAverOut());
            np.getMicroCluster().setTime(timestamp);
            np.getMicroCluster().setCategory("nov");
            np.getMicroCluster().setLabelClass("NP" + Integer.toString(this.getNPs().size() + 1));
            novelties.add(np);
        }
        
        this.getModel().put("NP" + Integer.toString(this.getNPs().size() + 1), novelties);
        this.writeCreateModelInfo(timestamp, novelties);
        this.addNPs(timestamp);
        textoArq = textoArq.concat("ThinkingNov: " + "Novelty " + "NP" + Integer.toString(this.getNPs().size() + 1) + " - " + novelties.size() + " micro-clusters" + "\n");
        fileOut.write(textoArq+"\n");
        System.out.println("ThinkingNov: " + "Novelty " + "NP" + Integer.toString(this.getNPs().size() + 1) + " - " + novelties.size() + " micro-clusters");
    }
    
    /**
     * Updates the models adding the new micro-clusters
     *
     * @param modelUnk micro-grupo novidade
     * @param model modelo a ser atualizado
     * @param classLabel rotulo do modelo
     * @param category tipo de micro-grupo (ext, nov,...)
     */
    public void updateModel(MicroClusterBR modelUnk, String label, String category, int timestamp) {
        //update the decision model by adding new valid micro-clusters
        modelUnk.getMicroCluster().setCategory(category);
        modelUnk.getMicroCluster().setLabelClass(label);
        modelUnk.getMicroCluster().setTime(timestamp);
        modelUnk.calculateThreshold(mtxLabelsFrequencies, numberOfObservedExamples);
        model.get(label).add(modelUnk);
    }
    
    public List<Map.Entry<Integer, Instance>> getInstancesUsedToBuildNewMC(int[] removeExamples, int indexMC) {
        List<Map.Entry<Integer, Instance>> toClassify = new ArrayList<>();
        for (int indexForRemoving = 0; indexForRemoving < removeExamples.length; indexForRemoving++) {
            // mark the examples to be removed with the label -2
            if (removeExamples[indexForRemoving] == indexMC) {
                removeExamples[indexForRemoving] = -2;
                Instance inst = this.getShortTimeMemory().getData().get(indexForRemoving);
                int timestamp = this.getShortTimeMemory().getTimestamp().get(indexForRemoving)
                toClassify.add(Map.entry(timestamp, inst)); //add to classify
            }
        }
        return toClassify;
    }

    public void updateMicroClusterThresholds() {
        model.entrySet().forEach(entry -> {
            entry.getValue().stream().map(x -> x).forEach(mc -> {
                mc.calculateThreshold(mtxLabelsFrequencies, this.numberOfObservedExamples);
            });
        });
    }

    private void writeCreateModelInfo(int timestamp, ArrayList<MicroClusterBR> novelties) {
       try {
            FileWriter file = null;
            if(novelties.get(0).getMicroCluster().getLabelClass().equals("NP1")){
                file = new FileWriter(new File(outputDirectory+"/createModelInfo.csv"), false);
                file.write("timestamp,evaluationWindow,nExamples,radius,silhouette,threshold,averOut,label" +"\n");
            }else{
               file = new FileWriter(new File(outputDirectory+"/createModelInfo.csv"), true);
            }
            
            int evaluationWindow = (int) Math.ceil(timestamp / this.evaluationWindowSize);
            
            for (MicroClusterBR np : novelties) {
            file.write(timestamp + ","+
                 evaluationWindow + ","+
                 np.getMicroCluster().getN()+ ","+
                 np.getMicroCluster().getRadius()+ ","+
                 np.clusterSilhouette(novelties)+ ","+
                 np.getThreshold()+ ","+
                 np.getAverOut()+ ","+
                 np.getMicroCluster().getLabelClass()+ "\n");
             }
            
            file.close();
        } catch (IOException ex) {
            Logger.getLogger(Model.class.getName()).log(Level.SEVERE, null, ex);
            System.err.println("Falha no arquivo createModelInfo.csv");
        }
    }

    public void updateCardinality(int windowSize) throws IOException {
        double contWindowObservedExamples = 0;
        double sum = 0;
        for(int i = this.Zall.size() - windowSize; i < this.Zall.size(); i++){
            if(!Zall.get(i).contains("unk")){
                sum += Zall.get(i).size();
                contWindowObservedExamples += 1;
            }
        }
        this.currentCardinality = sum / contWindowObservedExamples;
    }
    
    public void resetMtxLabelFrequencies(int windowSize) throws IOException {
        this.mtxLabelsFrequencies.keySet().forEach(cord -> {
            this.mtxLabelsFrequencies.put(cord, 0);
        });
        int contWindowsobservedExamples = 0;
        for(int i = this.Zall.size() - windowSize; i < this.Zall.size(); i++){
            if(!Zall.get(i).contains("unk")){
                this.updateMtxFrequencies(Zall.get(i));
                contWindowsobservedExamples++;
            }
        }
        FileWriter file = new FileWriter(new File(this.outputDirectory+"/infoResetMtx.csv"), true);
        for (Map.Entry<String, ArrayList<MicroClusterBR>> entry : model.entrySet()) {
            ArrayList<MicroClusterBR> set = entry.getValue();
            for (MicroClusterBR mc : set) {
                file.write(mc.getThreshold() + ",");
                mc.calculateThreshold(mtxLabelsFrequencies, contWindowsobservedExamples);
                file.write(mc.getThreshold() + "\n");
            }
        }
        file.close();
    }

    /**
     * Represents a Novelty Pattern found by the model
     */
    public class NoveltyPattern{
        private String label;
        private HashMap<String, int[][]> mtxCorr;
        private int timeStampCriation;
        private int timeStampAssociation;
        private Class associateClass;
        
        
        public NoveltyPattern(String label,  int timestamp) {
            this.label = label;
            this.setTimeStampCriation(timestamp);
            this.mtxCorr = new HashMap<>();
            for (Class c : getClasses()) {
                if(c.isNoveltyClass()){
                    String next = c.getLabel();
                    this.mtxCorr.put(next, new int[2][2]);
                }
            }
        }
        
        /**
         * Increments the NP
         * @param NP 
         */
        public void updateMtxCorr(Set<String> Z, Set<String> Y){
            for (Map.Entry<String, int[][]> entry : mtxCorr.entrySet()) {
                String key = entry.getKey();
                int[][] value = entry.getValue();
                
                if(Z.contains(this.getLabel()) && Y.contains(key)){
                    value[0][0] += 1;//a
                }else if(!Z.contains(this.getLabel()) && !Y.contains(key)){
                    value[1][1] += 1;//d
                }else if(Z.contains(this.getLabel()) && !Y.contains(key)){
                    value[1][0] += 1;//c
                }else {
                    value[0][1] += 1;//b
                }
            }
        }

        /**
         * @return the timeStampCriation
         */
        public int getTimeStampCriation() {
            return timeStampCriation;
        }

        /**
         * @param timeStampCriation the timeStampCriation to set
         */
        public void setTimeStampCriation(int timeStampCriation) {
            this.timeStampCriation = timeStampCriation;
        }

        /**
         * @return the associateClass
         */
        public Class getAssociateClass() {
            return associateClass;
        }

        /**
         * @param associateClass the associateClass to set
         */
        public void setAssociateClass(Class associateClass) {
            this.associateClass = associateClass;
        }

        /**
         * @return the label
         */
        public String getLabel() {
            return label;
        }

        /**
         * @return the timeStampAssociation
         */
        public int getTimeStampAssociation() {
            return timeStampAssociation;
        }

        /**
         * @param timeStampAssociation the timeStampAssociation to set
         */
        public void setTimeStampAssociation(int timeStampAssociation) {
            this.timeStampAssociation = timeStampAssociation;
        }
    }
    
    /**
     * Represents a problem class (novelty or known)
     */
    public class Class{
        private String label;
        private int timeStamp;
        private boolean noveltyClass;

        public Class(String label, int timeStamp, boolean noveltyClass) {
            this.label = label;
            this.timeStamp = timeStamp;
            this.noveltyClass = noveltyClass;
        }

        /**
         * @return the label
         */
        public String getLabel() {
            return label;
        }

        /**
         * @return the timeStamp
         */
        public int getTimeStamp() {
            return timeStamp;
        }

        /**
         * @return the noveltyClass
         */
        public boolean isNoveltyClass() {
            return noveltyClass;
        }

    }
    
    /**
     * Gets all label's classes of model
     * @return 
     */
    public Set<String> getAllLabel(){
        Set<String> set = new HashSet<String>();
        for (Class classe : getClasses()) {
            set.add(classe.getLabel());
        }
        return set;
    }
    
    /**
     * Gets a class given a label
     * @param label
     * @return 
     */
    public Class getClass(String label){
        for (Class c : this.getClasses()) {
            if(c.getLabel().equals(label))
                return c;
        }
        return null;
    }
     /**
     * Gets a Novelty Pattern given a label
     * @param label
     * @return 
     */
    public NoveltyPattern getNP(String label){
        for (NoveltyPattern NP : getNPs()) {
            if(NP.getLabel().equals(label)){
                return NP;
            }
        }
        return null;
    }
    
    /**
     * Associates NP to the problem classes
     * @param windowSize
     * @param timestamp
     * @param measure
     * @throws IOException 
     */
    public void associatesNPs(int windowSize, int timestamp, String measure) throws IOException{
        //If the model has built NPs
        if(!this.NPs.isEmpty()){
            //Fills the contince matrices
            for (int i = this.Yall.size() - windowSize; i < this.Yall.size(); i++) {
                for (int j = 0; j < this.getNPs().size(); j++) {
                    try{
                        if(!Zall.get(i).contains("unk"))
                            this.getNPs().get(j).updateMtxCorr(Zall.get(i), Yall.get(i));
                    }catch(Exception e){
                        System.out.println("");
                    }
                }
            }
            
            for (int i = 0; i < this.getNPs().size(); i++) {
                //If the NP hasn't had an association yet
                if(getNPs().get(i).getAssociateClass() == null){
                    HashMap<String, Float> scores = new HashMap<String, Float>();
                    this.getPnInfo().write("======== "+measure + " " + getNPs().get(i).label + " timestamp = " + timestamp + "========="+"\n");
                    
                    for (Map.Entry<String, int[][]> entry : this.getNPs().get(i).mtxCorr.entrySet()) {
                        String key = entry.getKey();
                        int[][] value = entry.getValue();
                            int a = value[0][0];
                            int b = value[0][1];
                            int c = value[1][0];
                            int d = value[1][1];
                            
                            float score = 0;
                            switch(measure){
                                case "JI":
                                    score = this.testJ(a, b, c, d);
                                    break;
                                case "FM":
                                    score = this.testF1(a, b, c, d);
                                    break;
                                case "X2":
                                    score = this.testChiSquared(a, b, c, d);
                                    break;
                                default:
                                    System.out.println("Invalid measure");
                                    break;
                            }
                            scores.put(key, score);
                            this.getPnInfo().write("C: " + key + " - a: " + a + " b: " + b + " c: " + c + " d: " + d + 
                                    " score: " + score + "\n");
                    }

                    String labelMax = "";
                    double scoreMax = 0;

                    for (Map.Entry<String, Float> entry : scores.entrySet()) {
                        String key = entry.getKey();
                        float value = entry.getValue();
                        if(value > scoreMax){
                            labelMax = key;
                            scoreMax = value;
                        }
                    }
                    float x = 0;
                    if(measure.equals("X2")) //If chi-squared test, the score must be greater then a critical value
                        x = (float) 3.841; /*p-value*/
                    if(scoreMax > x ){
                        Class c = this.getClass(labelMax);
                        getNPs().get(i).setAssociateClass(c);
                        this.getNPs().get(i).setTimeStampAssociation(timestamp);
                        this.getPnInfo().write(getNPs().get(i).getLabel() + " --> " + c.getLabel() + " - t: " + timestamp + " - score: " + scoreMax +"\n");
                    }else{ //if the score is zero then it doesn't have information to assossiate the NP to a class
                        System.out.println(getNPs().get(i).getLabel() + " had no class associated with it at this window");
                        this.getPnInfo().write(getNPs().get(i).getLabel() + " had no class associated with it until the timestamp " + timestamp + "\n");
                    }
                    this.getPnInfo().write("====================================" + "\n");
                }
            }
        }
        
    }
    /**
     * Associates NP to the problem classes (comparable measures)
     * @param windowSize
     * @param timestamp
     * @throws IOException 
     */
    public void associatesNPs(int windowSize, int timestamp) throws IOException{
        //If the model has built NPs
        if(!this.NPs.isEmpty()){
            //Fill the unknown cell of the contigence matrices 
            for (int i = this.Yall.size() - windowSize; i < this.Yall.size(); i++) {
                for (int j = 0; j < this.getNPs().size(); j++) {
                    if(!Zall.get(i).contains("unk"))
                        this.getNPs().get(j).updateMtxCorr(Zall.get(i), Yall.get(i));
                }
            }
            
            for (int i = 0; i < this.getNPs().size(); i++) {
                //If the NP hasn't had an association yet
                if(getNPs().get(i).getAssociateClass() == null){
                    HashMap<String, Float> scoresJI = new HashMap<String, Float>();
                    HashMap<String, Float> scoresX2 = new HashMap<String, Float>();
                    HashMap<String, Float> scoresFM = new HashMap<String, Float>();
                    this.getPnInfo().write("========"+getNPs().get(i).label + " timestamp = " + timestamp + "========="+"\n");

                    for (Map.Entry<String, int[][]> entry : this.getNPs().get(i).mtxCorr.entrySet()) {
                        String key = entry.getKey();
                        int[][] value = entry.getValue();
    //                    if(this.getNoveltyClasses().get(key).getTimeStamp() <= (timestamp - windowSize)){
                            int a = value[0][0];
                            int b = value[0][1];
                            int c = value[1][0];
                            int d = value[1][1];
                            
                            //get association by fmeasured
                            float scoreJI = this.testJ(a, b, c, d);
                            float scoreFM = this.testF1(a, b, c, d);
                            float scoreX2 = this.testChiSquared(a, b, c, d);
                            scoresJI.put(key, scoreJI);
                            scoresFM.put(key, scoreFM);
                            scoresX2.put(key, scoreX2);
                            this.getPnInfo().write("C: " + key + " - a: " + a + " b: " + b + " c: " + c + " d: " + d + 
                                    " scoreJI: " + scoreJI + " scoreFM: " + scoreFM+" scoreX2: " + scoreX2+ "\n");
                    }

                    String labelMaxJI = "";
                    String labelMaxFM = "";
                    String labelMaxX2 = "";
                    double scoreMaxJI = 0;
                    double scoreMaxX2 = 0;
                    double scoreMaxFM = 0;

                    for (Map.Entry<String, Float> entry : scoresJI.entrySet()) {
                        String key = entry.getKey();
                        float value = entry.getValue();
                        if(value > scoreMaxJI){
                            labelMaxJI = key;
                            scoreMaxJI = value;
                        }
                    }
                    for (Map.Entry<String, Float> entry : scoresFM.entrySet()) {
                        String key = entry.getKey();
                        float value = entry.getValue();
                        if(value > scoreMaxFM){
                            labelMaxFM = key;
                            scoreMaxFM = value;
                        }
                    }
                    for (Map.Entry<String, Float> entry : scoresX2.entrySet()) {
                        String key = entry.getKey();
                        float value = entry.getValue();
                        if(value > scoreMaxX2){
                            labelMaxX2 = key;
                            scoreMaxX2 = value;
                        }
                    }

                    if(scoreMaxJI > 0 /*p-value*/){
//                    if(scoreMax > 3.841 /*p-value*/){
                        Class c = this.getClass(labelMaxJI);
//                        NPs.get(i).setAssociateClass(c);
//                        this.getPnInfo().write(NPs.get(i).getLabel() + " --> " + c.getLabel() + " - t: " + timestamp + " - score: " + scoreMax +"\n");
                        this.getPnInfo().write("JI - " + getNPs().get(i).getLabel() + " --> " + c.getLabel() + " - t: " + timestamp + " - score: " + scoreMaxJI +"\n");
                    }else{
                        System.out.println("JI - " + getNPs().get(i).getLabel() + " não foi associado a nenhuma classe nesta janela");
                        this.getPnInfo().write("JI - " + getNPs().get(i).getLabel() + " não foi associado a nenhuma classe no timestamp: " + timestamp + "\n");
                    }
                    if(scoreMaxFM > 0 /*p-value*/){
//                    if(scoreMax > 3.841 /*p-value*/){
                        Class c = this.getClass(labelMaxFM);
//                        NPs.get(i).setAssociateClass(c);
                        this.getPnInfo().write("FM - " + getNPs().get(i).getLabel() + " --> " + c.getLabel() + " - t: " + timestamp + " - score: " + scoreMaxFM +"\n");
                    }else{
                        System.out.println("FM - " + getNPs().get(i).getLabel() + " não foi associado a nenhuma classe nesta janela");
                        this.getPnInfo().write("FM - " + getNPs().get(i).getLabel() + " não foi associado a nenhuma classe no timestamp: " + timestamp + "\n");
                    }
//                    if(scoreMax > 0 /*p-value*/){
                    if(scoreMaxX2 > 3.841 /*p-value*/){
                        Class c = this.getClass(labelMaxX2);
//                        NPs.get(i).setAssociateClass(c);
                        this.getPnInfo().write("X2 - " + getNPs().get(i).getLabel() + " --> " + c.getLabel() + " - t: " + timestamp + " - score: " + scoreMaxX2 +"\n");
                    }else{
                        System.out.println("X2 - " +getNPs().get(i).getLabel() + " não foi associado a nenhuma classe nesta janela");
                        this.getPnInfo().write("X2 - " +getNPs().get(i).getLabel() + " não foi associado a nenhuma classe no timestamp: " + timestamp + "\n");
                    }
                    this.getPnInfo().write("====================================" + "\n");
                }
            }
        }
        
    }
    
     /**
     * Calculates, based on a contingency table, the F1 value to create associations between NPs and Classes
     * @param a
     * @param b
     * @param c
     * @param d
     * @return 
     */
    private float testF1(int a, int b, int c, int d) {
        float fmeasure = 0;
        if(a > 0 && (b > 0 || c > 0)){
            float pr = ((float) a / ((float)(a+b)));
            float re = ((float) a / ((float)(a+c)));
            fmeasure = 2 * ((float)(pr*re) / ((float)(pr+re)));
        }
        return fmeasure;
    }
     /**
     * Calculates, based on a contingency table, the jaccard similarity coefficient value to create associations between NPs and Classes
     * @param a
     * @param b
     * @param c
     * @param d
     * @return 
     */
    private float testJ(int a, int b, int c, int d) {
        float j = 0;
        if(a > 0 && (b > 0 || c > 0)){
            j = (float)a / (float)(a + b + c);
        }
        return j;
    }
    
    /**
     * Chi-Square Test - Statistical Independence between NPs and classes
     * @param a
     * @param b
     * @param c
     * @param d
     * @return 
     */
    private float testChiSquared(int a, int b, int c, int d) {
        float x = 0;
        if(a > 0 ||  c > 0){
            x = (float) ((float)(Math.pow((a*d - b*c),2) * (a+b+c+d)) / ((float)((a+b) * (c+d) * (b+d) * (a+c))));
        }
        return x;
    }
    
    /**
     * Adds true labels and predict labels in models' memory
     * @param Y
     * @param Z 
     */
    public void addPrediction(Set<String> Y, Set<String> Z) {
        this.Yall.add(Y);

        //if a example have been considered unknown, the Z is null and a new set is created with a element 'unk'
        if(Z!=null){
            this.Zall.add(Z);
        }else{
            Set<String> set = new HashSet<String>();
            set.add("unk");
            this.Zall.add(set);
        }
    }
    
    public void removerUnknown(Set<String> labelSet) {
        this.classifiedUnk.add(labelSet);
    }
    
   public void addNPs(int timestamp){
       int nPn = this.getNPs().size()+1;
       NoveltyPattern NP = new NoveltyPattern("NP"+nPn, timestamp);
       this.getNPs().add(NP);
       this.timestampNP.add(timestamp);
   }
   

    /**
     * @return the model
     */
    public HashMap<String, ArrayList<MicroClusterBR>> getModel() {
        return model;
    }

    /**
     * @param model the model to set
     */
    public void setModel(HashMap<String, ArrayList<MicroClusterBR>> model) {
        this.model = model;
    }

    /**
     * @return the evaluatedExamples
     */
    public int getEvaluatedExamples() {
        return evaluatedExamples;
    }
    
     public void inicialize(String outputDir) throws IOException {
        this.model = new HashMap<String, ArrayList<MicroClusterBR>>();
        this.mtxLabelsFrequencies = new HashMap<>();
        this.mtxProbabilities = new HashMap<>();
        this.NPs = new ArrayList<>();
        this.timestampNP = new ArrayList<>();
        this.Classes = new ArrayList<>();
        this.Yall = new ArrayList<>();
        this.Zall = new ArrayList<>();
        this.setPnInfo(outputDir);
        this.outputDirectory = outputDir;
    }

    public ArrayList<Set<String>> getYall(){
        return this.Yall;
    }
    
    public ArrayList<Set<String>> getZall(){
        return this.Zall;
    }
    
    /**
     * Replaces Novelty Patterns for its associated novelty class, it happens to calculate the evalution measures.
     * @param pred 
     */
    public void noveltyPatternsToClassesPrediction(Set<String> pred){
        Object[] predAux = pred.toArray();
        for (int i = 0; i < predAux.length; i++) {
            String aux = String.valueOf(predAux[i]);
            if(aux.contains("NP")){
                try{ //Try to replace the NP for its associated class
                    String teste = this.getNP(aux).getAssociateClass().getLabel();
                    pred.remove(aux);
                    pred.add(teste);
                }catch(NullPointerException e){ //
                    System.out.println("NP"+aux+" hasn't been any associated class");
                }
            }
        }
    }
    
    /**
     * Verifies when a new class emerges
     * @param labels
     * @param timestamp 
     */
    public void verifyConceptEvolution(Set<String> labels, int timestamp) {
        for (Iterator<String> iterator = labels.iterator(); iterator.hasNext();) {
            String next = iterator.next();
            
            if(this.getClass(next) == null){
//                if(next.equals("2 ") || next.equals("9 ") || next.equals("4 ") || next.equals(""))
//                    System.out.println("");
                System.out.println("**** Emergin class *****");
                System.out.println("Novelty Class: " + next);
                Class noveltyClass = new Class(next, timestamp, true);
                this.getClasses().add(noveltyClass);
                //Creates a new correlation matrix for each NP which has none associated class
                for (int i = 0; i < this.getNPs().size(); i++) {
                    this.getNPs().get(i).mtxCorr.put(next, new int[2][2]);
                }
            }
            
        }
    }

    /**
     * @return the Classes
     */
    public ArrayList<Class> getClasses() {
        return Classes;
    }

    /**
     * @param Classes the Classes to set
     */
    public void setClasses(Set<String> Classes) {
        for (Iterator<String> iterator = Classes.iterator(); iterator.hasNext();) {
            String next = iterator.next();
            Class aClass = new Class(next, 1, false);
            this.Classes.add(aClass);
        }
    }
    
//    /**
//     * Writes in a file the Novelty Patterns and what class it be associated.
//     * @param outputFile
//     * @throws IOException 
//     */
//    public void writeNPInfo(String outputFile) throws IOException{
//        FileWriter fileWriter = new FileWriter(new File(outputFile + "/NPInfo.txt"), false);
//        for (NoveltyPattern NP : NPs) {
//            try{
//                fileWriter.write(NP.getLabel() + " = " + NP.getAssociateClass().getLabel() + "\n");
//            }catch(Exception e){
//                fileWriter.write(NP.getLabel() + " não foi vinculado a nenhum classe" + "\n");
//                System.out.println(NP.getLabel() + " não foi vinculado a nenhum classe");
//            }
//        }
//        fileWriter.close();
//    }

    /**
     * @return the pnInfo
     */
    public FileWriter getPnInfo() {
        return pnInfo;
    }

    /**
     * @param outputDir
     * @throws java.io.IOException
     */
    public void setPnInfo(String outputDir) throws IOException {
        this.pnInfo = new FileWriter(new File(outputDir + "/NPinfo.txt"), false);
    }

    /**
     * @return the NPs
     */
    public ArrayList<NoveltyPattern> getNPs() {
        return NPs;
    }

    /**
     * @return the stm
     */
    public ShortTimeMemory getShortTimeMemory() {
        return shortTimeMemory;
    }
    
    /**
     * Deletes obsolete examples from the short-time-memory
     * @param windowSize
     * @param timestamp
     * @param fileOut
     * @param flagFinal if the end of stream
     * @throws IOException
     */
    public void clearSortTimeMemory(int windowSize, int timestamp, FileWriter fileOut, boolean flagFinal) throws IOException {
        int deletedExamples = 0;
        for (int i = 0; i < this.getShortTimeMemory().getTimestamp().size(); i++) {
            if ((this.getShortTimeMemory().getTimestamp().get(i) < (timestamp - windowSize)) || flagFinal) {
                String textoArq = "Removed examples from short-time-memory: " + this.getShortTimeMemory().getTimestamp().get(i) + 
                        "\t True labels: " + DataSetUtils.getLabels(this.getShortTimeMemory().getData().get(i));
//                System.out.println(textoArq);s
                fileOut.write(textoArq);
                fileOut.write("\n");
//                av.updateExampleBasedMeasures(Z, Y);
//                av.updateLabelBasedMeasures(Z, Y);
                this.getShortTimeMemory().remove(i);
                i--;
                deletedExamples++;
            }
        }
        this.getShortTimeMemory().setQtdeExDeleted(this.getShortTimeMemory().getQtdeExDeleted()+deletedExamples);
        System.out.println("Timestamp: " + timestamp + " - number of removed examples: " + deletedExamples);
    }

}
