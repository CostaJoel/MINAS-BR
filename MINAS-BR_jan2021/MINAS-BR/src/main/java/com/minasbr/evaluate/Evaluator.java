/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.minasbr.evaluate;

import com.minasbr.br.Model;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Prediction;
import com.minasbr.dataSource.DataSet;
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
import java.util.Set;
import meka.core.ThresholdUtils;
import com.minasbr.utils.Voting;

/**
 *
 * @author bioinfo02
 */
public class Evaluator {

    private int nLabels;
    private String classifier;
    private int contEval = 0;
    private HashMap<String, int[][]> consufionMtx;
     
    
    private ArrayList<Float> f1m = new ArrayList<>();
    private ArrayList<Float> f1M = new ArrayList<>();
    private ArrayList<Float> prM = new ArrayList<>();
    private ArrayList<Float> reM = new ArrayList<>();
    private ArrayList<Float> prm = new ArrayList<>();
    private ArrayList<Float> rem = new ArrayList<>();
    private int TP = 0;
    private int FP = 0;
    private int TN = 0;
    private int FN = 0;
    
    private ArrayList<Float> f1 = new ArrayList<>();
    private ArrayList<Float> sub_acc = new ArrayList<>();
    private ArrayList<Float> hl = new ArrayList<>();
    private ArrayList<Float> pr = new ArrayList<>();
    private ArrayList<Float> re = new ArrayList<>();
    private float sumF1 = 0;
    private float sumSubAcc = 0; 
    private float sumHL = 0; 
    private float sumPr = 0; 
    private float sumRe = 0; 
    
    public Evaluator(int nLabels, Set<String> knownClasses, String classifier) {
        this.nLabels = nLabels;
        this.classifier = classifier;
        this.consufionMtx = new HashMap<String, int[][]>();
        for (Iterator<String> iterator = knownClasses.iterator(); iterator.hasNext();) {
            String next = iterator.next();
            this.consufionMtx.put(next, new int[2][3]);
        }
    }
    
    /**
     * Get example true label
     * @param labels labels
     * @return bipartition vector
     */
    public int[] getBipartition(String labels){
        char[] Z_char = labels.toCharArray();
        int[] Z = new int[this.getnLabels()];
        for (int j = 1; j < Z_char.length-1; j++) {
            if(Z_char[j] != ' ' && Z_char[j] != ','){
                Z[Integer.parseInt(String.valueOf(Z_char[j]))] = 1;
            }
        }
        return Z;
    }

    
    
    /**
     * Seleciona os rótulos mais relevantes através de um limiar
     * @param theashold
     * @param distribution
     * @return 
     */
    public int[] threasholding(double theashold, double[] distribution){
        int[] Z = new int[getnLabels()];
        for (int i = 0; i < distribution.length; i++) {
            if(distribution[i] > theashold){
                Z[i] = 1;
            }
        }
        return Z;
    }

    /**
     * Atualiza as medidas de avaliação baseadas em exemplos
     *
     * @param Z rótulos preditos
     * @param Y rótulos reais
     */
    public void updateExampleBasedMeasures(Set<String> Z, Set<String> Y) {
        //Did a new class emerge?
        if(!this.getConsufionMtx().keySet().containsAll(Y)){
            for (Iterator<String> iterator = Y.iterator(); iterator.hasNext();) {
                String next = iterator.next();
                if(!this.getConsufionMtx().keySet().contains(next)){
                    getConsufionMtx().put(next, new int[2][3]);
                }
            }
        }
        this.setContEval(this.getContEval()+1); //contrala a quantidade de exemplos que estão sendo avaliado na janela
        Set<String> intersection = new HashSet<String>();
        Set<String> union = new HashSet<String>();
        Set<String> delta = new HashSet<String>();

        intersection.addAll(Z);
        intersection.retainAll(Y);
        union.addAll(Z);
        union.addAll(Y);
        
        //Symmetric difference
        delta.addAll(union);
        delta.removeAll(intersection);
        float auxPr = getSumPr() + ((float) intersection.size() / Z.size());
        if(Float.isNaN(auxPr)){
            auxPr = 0;
            this.setSumPr(auxPr);
        }else
            this.setSumPr(auxPr);
        setSumRe(getSumRe() + ((float) intersection.size() / Y.size()));
        setSumF1(getSumF1() + (float) (((float)2 * intersection.size()) / ((float)Z.size() + Y.size())));
        if(Z.containsAll(Y) && Y.containsAll(Z)){
            this.setSumSubAcc((float) this.getSumSubAcc() + 1);
        }
        setSumHL((float)getSumHL() + (delta.size() / this.getConsufionMtx().size())); //confusionMtx().size() traz a 
                                                                                      //quantidade de classes conhecidas até o momento
    }
    
    
    /**
     * Updates confusion matrices
     * @param Y
     * @param Z
     */
    public void updateLabelBasedMeasures(Set<String> Z, Set<String> Y){
        for (Map.Entry<String, int[][]> entry : getConsufionMtx().entrySet()) {
            String key = entry.getKey();
            int[][] value = entry.getValue();
            if(Z.contains(key) && Y.contains(key)){ //TP
                value[0][0] += 1;
            }else if(Z.contains(key) && !Y.contains(key)){//FP
                value[0][1] += 1;
            }else if(!Z.contains(key) && Y.contains(key)){ //FN
                value[1][0] += 1;
            }else if(!Z.contains(key) && !Y.contains(key)){ //TN
                value[1][1] += 1;
            }
        }
    }
    
    /**
     * Calculates the measures' average by window
     * @param timestamp
     */
    public void calculateWindowMeasures() {
        //Example based measures
        float aux = (float) this.getSumF1() / this.getContEval();
        if(Float.isNaN(aux)){
            aux = 0;
        }
        this.getF1().add(aux);
        aux = (float)this.getSumHL() / this.getContEval();
        if(Float.isNaN(aux)){
            aux = 0;
        }
        this.getHl().add(aux);
        aux=(float)this.getSumSubAcc() / this.getContEval();
        if(Float.isNaN(aux)){
            aux = 0;
        }
        this.getSub_acc().add(aux);
        aux=(float) this.getSumPr() / this.getContEval();
        if(Float.isNaN(aux)){
            aux = 0;
        }
        this.getPr().add(aux);
        aux=(float) this.getSumRe() / (float)this.getContEval();
         if(Float.isNaN(aux)){
            aux = 0;
        }
        this.getRe().add(aux);
        
        float pr = 0;
        float re = 0;
        //macro
        for (Map.Entry<String, int[][]> entry : getConsufionMtx().entrySet()) {
            int[][] value = entry.getValue();
            float pre = ((float) value[0][0] / (value[0][0]+value[0][1]));
            if(Float.isNaN(pre)){
                pre = 0;
            }
            pr += pre;
            float rec = ((float) value[0][0] / (value[0][0]+value[1][0]));
            if(Float.isNaN(rec)){
                rec = 0;
            }
            re += rec;
        }
        float auxPrM = ((float) pr / this.getConsufionMtx().size());
        if(Float.isNaN(auxPrM)){
            auxPrM = 0;
            getPrM().add(auxPrM);
        }else{
            getPrM().add(auxPrM);
        }
        float auxReM = (float) (re / this.getConsufionMtx().size());
        if(Float.isNaN(auxReM)){
            auxReM = 0;
            getReM().add(auxReM);
        }else{
            getReM().add(auxReM);
        }
        float auxF1M = (float)(((float)2*((float)auxPrM * auxReM)) / ((float)auxPrM+auxReM));
        if(Float.isNaN(auxF1M)){
            auxF1M = 0;
            getF1M().add(auxF1M);
        }else{
            getF1M().add(auxF1M);
        }
        
        //micro
        for (Map.Entry<String, int[][]> entry : getConsufionMtx().entrySet()) {
            int[][] value = entry.getValue();
            setTP(getTP() + value[0][0]);
            setFP(getFP() + value[0][1]);
            setTN(getTN() + value[1][1]);
            setFN(getFN() + value[1][0]);
        }
        
        pr = (float)(getTP() / ((float) getTP() + getFP()));
        if(Float.isNaN(pr))
           pr = 0;
        re = (float)(getTP() / ((float)getTP() + getFN()));
        if(Float.isNaN(re))
            re = 0;
        aux = (float)(2*((float)((float)pr*re) / ((float)pr+re)));
        if(Float.isNaN(aux))
            aux = 0;
        getF1m().add(aux);
        getPrm().add(pr);
        getRem().add(re);
    }
    
    /**
     * Writes in fileOut.csv the measures over time
     * @param outputDirectory
     * @throws IOException 
     */
    public void writeMeasuresOverTime(String outputDirectory) throws IOException{
        FileWriter fileOut = new FileWriter(new File(outputDirectory + "/"+this.getClassifier()+"-Results.csv"), false);
        System.out.println("====================== Resultados por Janela ======================");
        System.out.println("Timestamp,F1,SA,HL,Pr,Re,F1m,F1M,Prm,PrM,Rem,ReM,unkR,unk");
        fileOut.write("Timestamp,F1,SA,HL,Pr,Re,F1m,F1M,Prm,PrM,Rem,ReM,unkR,unk\n");

        for (int i = 0; i < getF1m().size(); i++) {
            fileOut.write(i+1 + "," + getF1().get(i)+ "," + getSub_acc().get(i) + "," + getHl().get(i) + "," + getPr().get(i) + "," + getRe().get(i) + "," +
                    getF1m().get(i) + "," + getF1M().get(i) + ","+ getPrm().get(i) + ","+ getPrM().get(i) + ","+ getRem().get(i) + ","+ getReM().get(i) +
                    /*","+ getUnk().get(i) + "," + getUnk().get(i) +*/ "\n");
            System.out.println(i+1 + "," + getF1().get(i)+ "," + getSub_acc().get(i) + "," + getHl().get(i) + "," + getPr().get(i) + "," + getRe().get(i) + "," +
                    getF1m().get(i) + "," + getF1M().get(i) + ","+ getPrm().get(i) + ","+ getPrM().get(i) + ","+ getRem().get(i) + ","+ getReM().get(i) /*+
                    ","+ getUnk().get(i) + "," + getUnk().get(i)*/);
        }
        
        fileOut.close();
    }
    /**
     * Writes in fileOut.csv the measures over time
     * @param av
     * @param outputDirectory
     * @throws IOException 
     */
    public static void writeMeasuresOverTime(ArrayList<Evaluator> av, String outputDirectory) throws IOException{
        FileWriter fileOut = new FileWriter(new File(outputDirectory + "/"+"MeasuresOverTime.csv"), false);
        fileOut.write("Classifier,Timestamp,F1,SA,HL,Pr,Re,F1m,F1M,Prm,PrM,Rem,ReM\n");
        
        for (Evaluator ev : av) {
            for (int i = 0; i < ev.getF1m().size(); i++) {
                int timestamp = i+1;
                fileOut.write(ev.getClassifier() + "," + timestamp + "," + ev.getF1().get(i)+ "," + ev.getSub_acc().get(i) + "," + ev.getHl().get(i) + 
                        "," + ev.getPr().get(i) + "," + ev.getRe().get(i) + "," + ev.getF1m().get(i) + "," + ev.getF1M().get(i) + ","+
                        ev.getPrm().get(i) + ","+ ev.getPrM().get(i) + ","+ ev.getRem().get(i) + ","+ ev.getReM().get(i) +"\n");
            }
            if (ev.getClassifier().equals("MINAS-BR")) {
                EvaluatorBR unk = (EvaluatorBR)ev;
                for (int i = 0; i < unk.getUnkRM().size(); i++) {
                    int timestamp = i + 1;
                    fileOut.write("UnkRM," + timestamp + "," + unk.getUnkRM().get(i) + "," + unk.getUnkRM().get(i) + "," + unk.getUnkRM().get(i)
                            + "," + unk.getUnkRM().get(i) + "," + unk.getUnkRM().get(i) + "," + unk.getUnkRM().get(i) + "," + unk.getUnkRM().get(i) + ","
                            + unk.getUnkRM().get(i) + "," + unk.getUnkRM().get(i) + "," + unk.getUnkRM().get(i) + "," + unk.getUnkRM().get(i) + "\n");
                }
//                for (int i = 0; i < unk.getUnkRm().size(); i++) {
//                    int timestamp = i + 1;
//                    fileOut.write("UnkRm," + timestamp + "," + unk.getUnkRm().get(i) + "," + unk.getUnkRm().get(i) + "," + unk.getUnkRm().get(i)
//                            + "," + unk.getUnkRm().get(i) + "," + unk.getUnkRm().get(i) + "," + unk.getUnkRm().get(i) + "," + unk.getUnkRm().get(i) + ","
//                            + unk.getUnkRm().get(i) + "," + unk.getUnkRm().get(i) + "," + unk.getUnkRm().get(i) + "," + unk.getUnkRm().get(i) + "\n");
//                }
            }
        }
        fileOut.close();
    }

    /**
     * Writes the overall measures' averages  of all methods 
     *
     * @param listaAv
     * @param outputDirectory
     * @throws IOException
     */
    public static void writesAvgResults(ArrayList<Evaluator> listaAv, String outputDirectory) throws IOException {
        FileWriter avgResult = new FileWriter(new File(outputDirectory + "/avgResults.csv"), false);
        avgResult.write("Classifiers,F1,SA,HL,Pr,Re,F1m,F1M,Prm,PrM,Rem,ReM\n");

        for (Evaluator av : listaAv) {
            avgResult.write(av.getClassifier() + "," + av.getAvgF1() + "," + av.getAvgSA() + "," + av.getAvgHL() + "," + av.getAvgPr() + "," + av.getAvgRe() + "," +
                    av.getAvgF1m() + "," + av.getAvgF1M() + ","+ av.getAvgPrm() + ","+ av.getAvgPrM() + ","+ av.getAvgRem() + ","+ av.getAvgReM()+ "\n");
        }
        avgResult.close();
    }
    
    @Override
    public String toString() {
        return "================================================================"
                + "\n" + "Método = " + getClassifier() + "\n" + "Hamming-Loss = " + this.getAvgHL()+ "\n" + "F-Measure = " + this.getAvgF1() + 
                "\n" + "Subset-Accuracy = " + this.getAvgSA()+ "\n" + "Macro F-Measure = " + this.getAvgF1M() +
                "\n" + "Micro F-Measure = " + this.getAvgF1m();
    }
    
    /**
     * Convert a Prediction in bipartition 
     * @param yp
     * @param t
     * @return 
     */
    public int[] predictionToIntVector(Prediction yp, double t){
//        double[] y = new double[yp.numOutputAttributes()];
        int[] y = new int[yp.numOutputAttributes()];
        if(yp.numOutputAttributes() > 2){
            for (int j = 0; j < yp.numOutputAttributes(); j++) {
//                y[j] = (yp.getVote(j,0) >= yp.getVote(j,1)) ? 1 : 0;
                y[j] = (yp.getVote(j,1) >= t) ? 1 : 0;
            }
        }else{
            for (int i = 0; i < getnLabels(); i++) {
                y[i] = (yp.getVote(0,i) >= t) ? 1 : 0;
            }
        }
        return y;
    }
    
    /**
    * CalibrateThreshold - Calibrate a threshold using PCut: the threshold which results in the best approximation of the label cardinality of the training set.
    * @param	Y			labels
    * @param	LC_train	label cardinality of the training set
    */
   public static double calibrateThreshold(ArrayList<double[]> Y, double LC_train) { 

           if (Y.size() <= 0) 
                   return 0.5;

           int N = Y.size();
           ArrayList<Double> big = new ArrayList<Double>();
           for(double y[] : Y) {
                   for (double y_ : y) {
                           big.add(y_);
                   }
           }
           Collections.sort(big);

           int i = big.size() - (int)Math.round(LC_train * (double)N);

           if (N == big.size()) { // special cases
                   if (i+1 == N) // only one!
                           return (big.get(N-2)+big.get(N-1)/2.0);
                   if (i+1 >= N) // zero!
                           return 1.0;
                   else
                       return Math.max(((double)(big.get(i)+big.get(i+1))/2.0), 0.00001);
           }

           return Math.max(((double)(big.get(i)+big.get(Math.max(i+1,N-1))))/2.0 , 0.00001);
   }
    
   /**
    * Get the F1 average over the entire stream
    * @return 
    */
    public float getAvgF1(){
        float sumFM = 0;
        for (int i = 0; i < this.getF1().size(); i++) {
            sumFM += this.getF1().get(i);
        }
        float fm = sumFM/this.getF1().size();
        return fm;
    }
    
   /**
    * Get the HL average over the entire stream
    * @return 
    */
    public float getAvgHL(){
        float sumHL = 0;
        for (int i = 0; i < this.getHl().size(); i++) {
            sumHL += this.getHl().get(i);
        }
        float hl = sumHL/this.getHl().size();
        return hl;
    }
    
   /**
    * Get the SA average over the entire stream
    * @return 
    */
    public float getAvgSA(){
        float sumSA = 0;
        for (int i = 0; i < this.getSub_acc().size(); i++) {
            sumSA += this.getSub_acc().get(i);
        }
        float sub_acc = sumSA/this.getSub_acc().size();
        return sub_acc;
    }
    
   /**
    * Get the Pr average over the entire stream
    * @return 
    */
    public float getAvgPr(){
        float sumPr = 0;
        for (int i = 0; i < this.getPr().size(); i++) {
            sumPr += this.getPr().get(i);
        }
        float pr = sumPr/this.getPr().size();
        return pr;
    }
    
   /**
    * Get the Re average over the entire stream
    * @return 
    */
    public float getAvgRe(){
        float sumRe = 0;
        for (int i = 0; i < this.getRe().size(); i++) {
            sumRe += this.getRe().get(i);
        }
        float re = sumRe/this.getRe().size();
        return re;
    }
   /**
    * Get the Prm average over the entire stream
    * @return 
    */
    public float getAvgPrm(){
        float sumPr = 0;
        for (int i = 0; i < this.getPrm().size(); i++) {
            sumPr += this.getPrm().get(i);
        }
        float pr = sumPr/this.getPrm().size();
        return pr;
    }
    
   /**
    * Get the Re average over the entire stream
    * @return 
    */
    public float getAvgRem(){
        float sumRe = 0;
        for (int i = 0; i < this.getRem().size(); i++) {
            sumRe += this.getRem().get(i);
        }
        float re = sumRe/this.getRem().size();
        return re;
    }
    /**
    * Get the F1m average over the entire stream
    * @return 
    */
    public float getAvgF1m(){
        float sumFM = 0;
        for (int i = 0; i < this.getF1m().size(); i++) {
            sumFM += this.getF1m().get(i);
        }
        float fm = sumFM/this.getF1m().size();
        return fm;
    }
    
    /**
    * Get the PrM average over the entire stream
    * @return 
    */
    public float getAvgPrM(){
        float sumPr = 0;
        for (int i = 0; i < this.getPrM().size(); i++) {
            sumPr += this.getPrM().get(i);
        }
        float pr = sumPr/this.getPrM().size();
        return pr;
    }
    
   /**
    * Get the ReM average over the entire stream
    * @return 
    */
    public float getAvgReM(){
        float sumRe = 0;
        for (int i = 0; i < this.getReM().size(); i++) {
            sumRe += this.getReM().get(i);
        }
        float re = sumRe/this.getReM().size();
        return re;
    }
    /**
    * Get the F1M average over the entire stream
    * @return 
    */
    public float getAvgF1M(){
        float sumFM = 0;
        for (int i = 0; i < this.getF1M().size(); i++) {
            sumFM += this.getF1M().get(i);
        }
        float fm = sumFM/this.getF1M().size();
        return fm;
    }
    
    
    
    /**
     * Update measures using PCut threshold
     * @param pred classifier prediction
     * @param card window cardinality
     * @param trueLabels true labels
     */
    public void updateMeasuresWithTresholding(ArrayList<Prediction> pred, double card, ArrayList<Set<String>> trueLabels){
        ArrayList<double[]> Y = new ArrayList<double[]>();
        for (int i = 0; i < pred.size(); i++) {
            double[] aux = new double[pred.get(i).numOutputAttributes()];
            for (int j = 0; j < pred.get(i).numOutputAttributes(); j++) {
                aux[j] = pred.get(i).getVote(j, 1);
            }
            Y.add(aux);
        }
        double t = 0;
//        try{
            t = calibrateThreshold(Y, card);
//        }catch(Exception e){
//            System.out.println("");
//        }
        HashSet<String> Z = new HashSet<>();
        for (int i = 0; i < Y.size(); i++) {
            int[] auxInt = new int[Y.get(i).length];
            for (int j = 0; j < auxInt.length; j++) {
                if(Y.get(i)[j] >= t)
                    Z.add(String.valueOf(j));
            }
            this.updateExampleBasedMeasures(Z, trueLabels.get(i));
            this.updateLabelBasedMeasures(Z, trueLabels.get(i));
        }
        this.calculateWindowMeasures();
    }
    /**
     * Update measures using PCut threshold
     * @param pred classifier prediction
     * @param card window cardinality
     * @param trueLabels true labels
     */
    public void updateMeasures(ArrayList<Prediction> pred, double card, ArrayList<Set<String>> trueLabels){
        ArrayList<double[]> Y = new ArrayList<double[]>();
        for (int i = 0; i < pred.size(); i++) {
            double[] aux = new double[pred.get(i).numOutputAttributes()];
            for (int j = 0; j < pred.get(i).numOutputAttributes(); j++) {
                aux[j] = pred.get(i).getVote(j, 1);
            }
            Y.add(aux);
        }
        double t = ThresholdUtils.calibrateThreshold(Y, card);
        
        for (int i = 0; i < pred.size(); i++) {
            HashSet<String> Z = new HashSet<>();
//            for (int j = 0; j < pred.get(i).numOutputAttributes(); j++) {
//                if(pred.get(i).getVote(j, 1) >= t){
//                    Z.add(""+j);
//                }
//            }
            if(Z.isEmpty()){
                for (int j = 0; j < pred.get(i).numOutputAttributes(); j++) {
                    if(pred.get(i).getVote(j, 1) > pred.get(i).getVote(j, 0)){
                        Z.add(""+j);
                    }
                }
            }
            this.updateExampleBasedMeasures(Z, trueLabels.get(i));
            this.updateLabelBasedMeasures(Z, trueLabels.get(i));
        }
        this.calculateWindowMeasures();
    }
    
    public void writesAvgResults(String outputDirectory) throws IOException {
        FileWriter avgResult = new FileWriter(new File(outputDirectory + "/avgResults.csv"), true);
        avgResult.write("Classifiers,F1,SA,HL,Pr,Re,F1m,F1M,Prm,PrM,Rem,ReM,unkRm,unkRM,unk\n");

        avgResult.write(this.getClassifier() + "," + this.getAvgF1() + "," + this.getAvgSA() + "," + this.getAvgHL() + "," + this.getAvgPr() +
            "," + this.getAvgRe() + "," + ","+ this.getAvgF1m() + "," + this.getAvgF1M() + ","+ 
            this.getAvgPrm() + ","+ this.getAvgPrM() + ","+ this.getAvgRem() + ","+this.getAvgReM() + ","+ this.getAvgUnkRm() + ","+ 
            this.getAvgUnkRM() + "," + this.getTotalUnk() + "\n");
        
        avgResult.close();
    }
    
    public void updateMeasuresThresholding(ArrayList<Prediction> pred, ArrayList<Set<String>> trueLabels,double[] cardinalities, FileWriter filePredictions){
        ArrayList<double[]> predictions = new ArrayList<double[]>();
        for (int i = 0; i < pred.size(); i++) {
            double[] aux = new double[pred.get(i).numOutputAttributes()];
            for (int j = 0; j < pred.get(i).numOutputAttributes(); j++) {
                aux[j] = pred.get(i).getVote(j, 1);
            }
            predictions.add(aux);
        }

        //thresholding
        double[] thresholds = null;
        try{
            thresholds = ThresholdUtils.calibrateThresholds(predictions, cardinalities);
        }catch(Exception e){
//            e.printStackTrace();
            System.out.println("[WARNING]: Yeast");
            for (int i = 0; i < pred.size(); i++) {
                HashSet<String> Z = new HashSet<>();
                for (int j = 0; j < pred.get(i).numOutputAttributes(); j++) {
                    if(pred.get(i).getVote(j,1) > pred.get(i).getVote(j,0))
                        Z.add(""+j);
                }
                System.out.println("True Labels: " + trueLabels.get(i).toString() + "\t Predicted: " + Z.toString());
                try{
                    filePredictions.write("True Labels: " + trueLabels.get(i).toString() + "\t Predicted: " + Z.toString()+ "\n");
                }catch(Exception ex){
                    ex.printStackTrace();
                    System.out.println("Erro na escrita das predições.");
                }
                this.updateExampleBasedMeasures(Z, trueLabels.get(i));
                this.updateLabelBasedMeasures(Z, trueLabels.get(i));
            }
            this.calculateWindowMeasures();
            return;
        }
        
        for (int i = 0; i < pred.size(); i++) {
            HashSet<String> Z = new HashSet<>();
            for (int j = 0; j < pred.get(i).numOutputAttributes(); j++) {
                if(pred.get(i).getVote(j,1) >= thresholds[j])
                    Z.add(""+j);
                
            }
            if(Z.isEmpty()){
                for (int j = 0; j < pred.get(i).numOutputAttributes(); j++) {
                    if(pred.get(i).getVote(j,1) > pred.get(i).getVote(j,0))
                        Z.add(""+j);
                }
            }
            System.out.println("True Labels: " + trueLabels.get(i).toString() + "\t Predicted: " + Z.toString());
            try{
                filePredictions.write("True Labels: " + trueLabels.get(i).toString() + "\t Predicted: " + Z.toString() + "\n");
            }catch(Exception ex){
                ex.printStackTrace();
                System.out.println("Erro na escrita das predições.");
            }
            this.updateExampleBasedMeasures(Z, trueLabels.get(i));
            this.updateLabelBasedMeasures(Z, trueLabels.get(i));
        }
        this.calculateWindowMeasures();
    }

    /**
     * @return the classifier
     */
    public String getClassifier() {
        return classifier;
    }

    /**
     * @param classifier the classifier to set
     */
    public void setClassifier(String classifier) {
        this.classifier = classifier;
    }

    /**
     * @return the consufionMtx
     */
    public HashMap<String, int[][]> getConsufionMtx() {
        return consufionMtx;
    }

    /**
     * @param consufionMtx the consufionMtx to set
     */
    public void setConsufionMtx(HashMap<String, int[][]> consufionMtx) {
        this.consufionMtx = consufionMtx;
    }

    

    /**
     * @param f1m the f1m to set
     */
    public void setF1m(ArrayList<Float> f1m) {
        this.f1m = f1m;
    }

    /**
     * @param f1M the f1M to set
     */
    public void setF1M(ArrayList<Float> f1M) {
        this.f1M = f1M;
    }

    /**
     * @param prM the prM to set
     */
    public void setPrM(ArrayList<Float> prM) {
        this.prM = prM;
    }

    /**
     * @param reM the reM to set
     */
    public void setReM(ArrayList<Float> reM) {
        this.reM = reM;
    }

    /**
     * @param prm the prm to set
     */
    public void setPrm(ArrayList<Float> prm) {
        this.prm = prm;
    }

    /**
     * @param rem the rem to set
     */
    public void setRem(ArrayList<Float> rem) {
        this.rem = rem;
    }

    /**
     * @return the TP
     */
    public int getTP() {
        return TP;
    }

    /**
     * @param TP the TP to set
     */
    public void setTP(int TP) {
        this.TP = TP;
    }

    /**
     * @return the FP
     */
    public int getFP() {
        return FP;
    }

    /**
     * @param FP the FP to set
     */
    public void setFP(int FP) {
        this.FP = FP;
    }

    /**
     * @return the TN
     */
    public int getTN() {
        return TN;
    }

    /**
     * @param TN the TN to set
     */
    public void setTN(int TN) {
        this.TN = TN;
    }

    /**
     * @return the FN
     */
    public int getFN() {
        return FN;
    }

    /**
     * @param FN the FN to set
     */
    public void setFN(int FN) {
        this.FN = FN;
    }

    /**
     * @param f1 the f1 to set
     */
    public void setF1(ArrayList<Float> f1) {
        this.f1 = f1;
    }

    /**
     * @return the sub_acc
     */
    public ArrayList<Float> getSub_acc() {
        return sub_acc;
    }

    /**
     * @param sub_acc the sub_acc to set
     */
    public void setSub_acc(ArrayList<Float> sub_acc) {
        this.sub_acc = sub_acc;
    }

    /**
     * @return the hl
     */
    public ArrayList<Float> getHl() {
        return hl;
    }

    /**
     * @param hl the hl to set
     */
    public void setHl(ArrayList<Float> hl) {
        this.hl = hl;
    }

    /**
     * @param pr the pr to set
     */
    public void setPr(ArrayList<Float> pr) {
        this.pr = pr;
    }

    /**
     * @param re the re to set
     */
    public void setRe(ArrayList<Float> re) {
        this.re = re;
    }

    /**
     * @return the sumF1
     */
    public float getSumF1() {
        return sumF1;
    }

    /**
     * @param sumF1 the sumF1 to set
     */
    public void setSumF1(float sumF1) {
        this.sumF1 = sumF1;
    }

    /**
     * @return the sumSubAcc
     */
    public float getSumSubAcc() {
        return sumSubAcc;
    }

    /**
     * @param sumSubAcc the sumSubAcc to set
     */
    public void setSumSubAcc(float sumSubAcc) {
        this.sumSubAcc = sumSubAcc;
    }

    /**
     * @return the sumHL
     */
    public float getSumHL() {
        return sumHL;
    }

    /**
     * @param sumHL the sumHL to set
     */
    public void setSumHL(float sumHL) {
        this.sumHL = sumHL;
    }

    /**
     * @return the sumPr
     */
    public float getSumPr() {
        return sumPr;
    }

    /**
     * @param sumPr the sumPr to set
     */
    public void setSumPr(float sumPr) {
        this.sumPr = sumPr;
    }

    /**
     * @return the sumRe
     */
    public float getSumRe() {
        return sumRe;
    }

    /**
     * @param sumRe the sumRe to set
     */
    public void setSumRe(float sumRe) {
        this.sumRe = sumRe;
    }

    /**
     * @return the nLabels
     */
    public int getnLabels() {
        return nLabels;
    }

    /**
     * @param nLabels the nLabels to set
     */
    public void setnLabels(int nLabels) {
        this.nLabels = nLabels;
    }

    /**
     * @return the f1m
     */
    public ArrayList<Float> getF1m() {
        return f1m;
    }

    /**
     * @return the f1M
     */
    public ArrayList<Float> getF1M() {
        return f1M;
    }

    /**
     * @return the prM
     */
    public ArrayList<Float> getPrM() {
        return prM;
    }

    /**
     * @return the reM
     */
    public ArrayList<Float> getReM() {
        return reM;
    }

    /**
     * @return the prm
     */
    public ArrayList<Float> getPrm() {
        return prm;
    }

    /**
     * @return the rem
     */
    public ArrayList<Float> getRem() {
        return rem;
    }

    /**
     * @return the f1
     */
    public ArrayList<Float> getF1() {
        return f1;
    }

    /**
     * @return the pr
     */
    public ArrayList<Float> getPr() {
        return pr;
    }

    /**
     * @return the re
     */
    public ArrayList<Float> getRe() {
        return re;
    }

    

    /**
     * @return the contEval
     */
    public int getContEval() {
        return contEval;
    }

    /**
     * @param contEval the contEval to set
     */
    public void setContEval(int contEval) {
        this.contEval = contEval;
    }

    public float getAvgUnkRm() {
        return 0;
    }

    public float getAvgUnkRM() {
        return 0;
    }

    public int getTotalUnk() {
        return 0;
    }

   
}
