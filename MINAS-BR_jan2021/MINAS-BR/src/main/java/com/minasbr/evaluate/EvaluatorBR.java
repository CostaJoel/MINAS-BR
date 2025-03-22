package com.minasbr.evaluate;

import com.minasbr.br.Model;
import com.minasbr.dataSource.DataSet;
import com.minasbr.dataSource.DataSetUtils;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

/**
 *
 * @author joel
 */
public class EvaluatorBR extends Evaluator{
    private int Lk;
    private int UNK = 0;
    private ArrayList<Integer> unk = new ArrayList<>();
    private ArrayList<Integer> removedUnk = new ArrayList<>();
    private ArrayList<Float> unkRm = new ArrayList<>();
    private ArrayList<Float> unkRM = new ArrayList<>();
    private Set<String> knownClasses;
    private int qtdeNP;
    
    public EvaluatorBR(int Lall, Set<String> knownClasses,  String classifier) {
        super(Lall, knownClasses, classifier);
        this.knownClasses = knownClasses;
        this.Lk = Lall;
    }
    
    /**
     * Updates label based measures
     * @param model
     * @param windowSize 
     */
    public void updateLabelBasedMeasure(Model model, int windowSize){
        
        //Creates new contingence matrices if a new class emerges
        if(!super.getConsufionMtx().keySet().containsAll(model.getAllLabel())){
            for (Iterator<String> iterator = model.getAllLabel().iterator(); iterator.hasNext();) {
                String next = iterator.next();
                if(!super.getConsufionMtx().keySet().contains(next)){
                    getConsufionMtx().put(next, new int[2][3]);
                }
            }
        }
        
        //Separates unknown examples and fill the confusion matrixes 
        for (int i = model.getYall().size()-windowSize; i < model.getYall().size(); i++) {
            Set<String> Z = model.getZall().get(i);
//            for (Iterator<String> iterator = Z.iterator(); iterator.hasNext();) {
//                String next = iterator.next();
//                if(next.contains("NP")){
//                    System.out.println("");
//                }
//            }
            Set<String> Y = model.getYall().get(i);
            if(!Z.contains("unk")){
                model.noveltyPatternsToClassesPrediction(Z);
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
        }
       float pr = 0;
        float re = 0;
//        float sumUnkRM = 0;
        //macro
        for (Map.Entry<String, int[][]> entry : getConsufionMtx().entrySet()) {
            int[][] value = entry.getValue();
            float pre = ((float) value[0][0] / ((float)value[0][0]+value[0][1])); //Precision
            if(Float.isNaN(pre)){
                pre = 0;
            }
            pr += pre;
            float rec = ((float) value[0][0] / ((float)value[0][0]+value[1][0])); //recall
            if(Float.isNaN(rec)){
                rec = 0;
            }
            re += rec;
//            float unkRM = ((float) value[0][2] / ((float) value[0][0] + value[0][1] +  value[0][2])); //unknwon ration
//            if(Float.isNaN(unkRM)){
//                unkRM = 0;
//            }
//            sumUnkRM += unkRM;
        }
        float auxPrM = ((float) pr / this.getConsufionMtx().size());
        if(Float.isNaN(auxPrM)){
            auxPrM = 0;
            getPrM().add(auxPrM);
        }else{
            getPrM().add(auxPrM);
        }
        float auxReM = (float) re / this.getConsufionMtx().size();
        if(Float.isNaN(auxReM)){
            auxReM = 0;
            super.getReM().add(auxReM);
        }else
            super.getReM().add(auxReM);
        float auxF1M = (float)(((float)2*((float)((float)auxPrM*auxReM) / ((float)auxPrM+auxReM))));
        if(Float.isNaN(auxF1M)){
            auxF1M = 0;
            getF1M().add(auxF1M);
        }else{
            getF1M().add(auxF1M);
        }
//        float auxUnkRM = ((float) sumUnkRM / this.getConsufionMtx().size());
//        if(Float.isNaN(auxUnkRM)){
//            auxF1M = 0;
//            this.getUnkRM().add(auxUnkRM);
//        }else{
//            this.getUnkRM().add(auxUnkRM);
//        }
        
        //micro
        for (Map.Entry<String, int[][]> entry : getConsufionMtx().entrySet()) {
            int[][] value = entry.getValue();
            setTP(getTP() + value[0][0]);
            setFP(getFP() + value[0][1]);
            setTN(getTN() + value[1][1]);
            setFN(getFN() + value[1][0]);
            this.setUNK(this.getUNK() + value[0][2]);
        }
        pr = (float)(getTP() / ((float) getTP() + getFP()));
        re = (float)(getTP() / ((float)getTP() + getFN()));
//        float unkRm = (float)(this.getUNK() / ((float)this.getUNK() + getTP() + getFP()));
        getF1m().add((float)(2*((float)((float)pr*re) / ((float)pr+re))));
        getPrm().add(pr);
        getRem().add(re);
//        getUnkRm().add(unkRm);
        updateUnknownRatio(model);
        this.removedUnk.add(model.getShortTimeMemory().getQtdeExDeleted());
        model.getShortTimeMemory().setQtdeExDeleted(0);
    }
    
    /**
     * Updates example based measures
     * @param model
     * @param windowSize 
     */
    public void updateExampleBasedMeasure(Model model, int windowSize){
        int contUnk = 0;
        for (int i = model.getYall().size()-windowSize; i < model.getYall().size(); i++) {
            Set<String> Z = model.getZall().get(i);
            Set<String> Y = model.getYall().get(i);
            Set<String> intersection = new HashSet<String>();
            Set<String> union = new HashSet<String>();
            Set<String> delta = new HashSet<String>();

            if(!Z.contains("unk")){
                super.setContEval(super.getContEval()+1);
                model.noveltyPatternsToClassesPrediction(Z);
//                for (Iterator<String> iterator = Z.iterator(); iterator.hasNext();) {
//                    String next = iterator.next();
//                    if(next.contains("NP")){
//                        System.out.println("");
//                    }
//                }
                intersection.addAll(Z);
                intersection.retainAll(Y);
                union.addAll(Z);
                union.addAll(Y);
                //Symmetric difference
                delta.addAll(union);
                delta.removeAll(intersection);
                
                setSumPr(getSumPr() + (float) intersection.size() / Z.size());
                setSumRe(getSumRe() + (float) intersection.size() / Y.size());
                setSumF1(getSumF1() + (float) (((float)(2 * intersection.size())) / ((float)(Z.size() + Y.size()))));
                if(Z.containsAll(Y) && Y.containsAll(Z)){
                    super.setSumSubAcc(getSumSubAcc() + 1);
                }
                setSumHL(getSumHL() + delta.size() / model.getClasses().size());
            }else{
               contUnk++;
            }
        }
        this.getUnk().add(contUnk);
        float f1 = (float)getSumF1()/super.getContEval();
        if(Float.isNaN(f1))
            f1 = 0;
        super.getF1().add(f1);
        
        float subAcc = (float)getSumSubAcc()/(float)super.getContEval();
        if(Float.isNaN(subAcc))
            subAcc = 0;
        super.getSub_acc().add(subAcc);
        
        float hl = (float) super.getSumHL()/(float)super.getContEval();
        if(Float.isNaN(hl))
            hl = 0;
        super.getHl().add(hl);
        
        float pr = (float) this.getSumPr()/(float)super.getContEval();
        if(Float.isNaN(pr))
            pr = 0;
        super.getPr().add(pr);
        
        float re = (float) this.getSumRe()/(float)super.getContEval();
        if(Float.isNaN(re))
            re = 0;
        super.getRe().add(re);
    }
    
    /**
     * Writes in fileOut.csv the measures over time
     * @param outputDirectory
     * @throws IOException 
     */
    @Override
    public void writeMeasuresOverTime(String outputDirectory) throws IOException{
        FileWriter fileOut = new FileWriter(new File(outputDirectory + "/"+this.getClassifier()+"-Results.csv"), false);
        System.out.println("====================== Resultados por Janela ======================");
        System.out.println("Timestamp,F1,SA,HL,Pr,Re,F1m,F1M,Prm,PrM,Rem,ReM,unkR,unk");
        fileOut.write("Timestamp,F1,SA,HL,Pr,Re,F1m,F1M,Prm,PrM,Rem,ReM,unkRM,unk,removedUnk\n");

        for (int i = 0; i < getF1m().size(); i++) {
            int t = i+1;
            fileOut.write(t + "," + getF1().get(i)+ "," + getSub_acc().get(i) + "," + getHl().get(i) + "," + getPr().get(i) + "," + getRe().get(i) + "," +
                    getF1m().get(i) + "," + getF1M().get(i) + ","+ getPrm().get(i) + ","+ getPrM().get(i) + ","+ getRem().get(i) + ","+ getReM().get(i) +
                    ","+ getUnkRM().get(i)+ "," + getUnk().get(i) + "," +
                    this.getRemovedUnk().get(i) + "\n");
            System.out.println(i+1 + "," + getF1().get(i)+ "," + getSub_acc().get(i) + "," + getHl().get(i) + "," + getPr().get(i) + "," + getRe().get(i) + "," +
                    getF1m().get(i) + "," + getF1M().get(i) + ","+ getPrm().get(i) + ","+ getPrM().get(i) + ","+ getRem().get(i) + ","+ getReM().get(i) +
                    ","+ getUnkRM().get(i)+ "," + getUnk().get(i));
        }
        
        fileOut.close();
    }

    public ArrayList<Integer> getRemovedUnk() {
        return removedUnk;
    }

    public void setRemovedUnk(ArrayList<Integer> removedUnk) {
        this.removedUnk = removedUnk;
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
        avgResult.write("Classifiers,F1,SA,HL,Pr,Re,F1m,F1M,Prm,PrM,Rem,ReM,unkRm,unkRM,unk\n");

        for (Evaluator av : listaAv) {
            avgResult.write(av.getClassifier() + "," + av.getAvgF1() + "," + av.getAvgSA() + "," + av.getAvgHL() + "," + av.getAvgPr() + "," + av.getAvgRe() + "," +
                    av.getAvgF1m() + "," + av.getAvgF1M() + ","+ av.getAvgPrm() + ","+ av.getAvgPrM() + ","+ av.getAvgRem() + ","+ av.getAvgReM() +
                    ","+ av.getAvgUnkRm() + ","+ av.getAvgUnkRM() + "," + av.getTotalUnk() + "\n");
        }
        avgResult.close();
    }

    /**
     * @return the UNK
     */
    public int getUNK() {
        return UNK;
    }

    /**
     * @param UNK the UNK to set
     */
    public void setUNK(int UNK) {
        this.UNK = UNK;
    }
    
    /**
     * @param unk the unk to set
     */
    public void setUnk(ArrayList<Integer> unk) {
        this.unk = unk;
    }
    
    /**
     * @return the unk
     */
    public ArrayList<Integer> getUnk() {
        return unk;
    }
    
     /**
     * @return the unkRm
     */
    public ArrayList<Float> getUnkRm() {
        return unkRm;
    }

    /**
     * @return the unkRmM
     */
    public ArrayList<Float> getUnkRM() {
        return unkRM;
    }
    
    /**
     * Get the total number of unknown examples of the stream
     * @return 
     */
    @Override
    public int getTotalUnk() {
        int totalUnk = 0;
        for (int i = 0; i < this.getUnk().size(); i++) {
            totalUnk += this.getUnk().get(i);
        }
        return totalUnk;
    }
    
    /**
    * Get the UnkRM average over the entire stream
    * @return 
    */
    @Override
    public float getAvgUnkRM(){
        float sumUnkRM = 0;
        for (int i = 0; i < this.getUnkRM().size(); i++) {
            sumUnkRM += this.getUnkRM().get(i);
        }
        float unkRM = sumUnkRM/this.getUnkRM().size();
        return unkRM;
    }
    
    /**
    * Get the UnkRm average over the entire stream
    * @return 
    */
    @Override
    public float getAvgUnkRm(){
        float sumUnkRm = 0;
        for (int i = 0; i < this.getUnkRm().size(); i++) {
            sumUnkRm += this.getUnkRm().get(i);
        }
        float unkRm = sumUnkRm/this.getUnkRm().size();
        return unkRm;
    }

    public void writeConceptEvolutionNP(Model model, String outputDirectory) throws IOException{
        //Novelty Pattern
        FileWriter NP_info = new FileWriter(new File(outputDirectory + "/NP-info.csv"), false);
        NP_info.write("Timestamp, NP, AssociatedClass, TimestampAssociation, windowAssociation \n");
        for (int i = 0; i < model.getNPs().size(); i++) {
            //Get the window which the NP was created
            String t = "";
            String label;
            String timeStampAssociation;
            if(model.getNPs().get(i).getAssociateClass() == null){
                label = "";
                timeStampAssociation = "";
            }else{
                label = model.getNPs().get(i).getAssociateClass().getLabel();
                timeStampAssociation = ""+model.getNPs().get(i).getTimeStampAssociation();
                int t1 = (int)Math.ceil(model.getNPs().get(i).getTimeStampAssociation() / (model.getZall().size() / this.getUnkRM().size()));
                t = ""+t1;
            }
            NP_info.write(model.getNPs().get(i).getTimeStampCriation() + "," + model.getNPs().get(i).getLabel() + 
                    "," + label + "," + timeStampAssociation + "," + t + "\n");
        }
        NP_info.close();
        
        //Concept Evolution
        FileWriter CE_info = new FileWriter(new File(outputDirectory + "/conceptEvolution-info.csv"), false);
        CE_info.write("Timestamp,window,label\n");
        for (int i = 0; i < model.getClasses().size(); i++) {
            if(model.getClasses().get(i).isNoveltyClass()){
                //Get the window which the new class appeared
                int t = (int)Math.ceil(model.getClasses().get(i).getTimeStamp() / ((int)(model.getZall().size() / this.getUnkRM().size())));
                CE_info.write(model.getClasses().get(i).getTimeStamp()+","+ t + "," + model.getClasses().get(i).getLabel() + "\n");
                System.out.println( model.getClasses().get(i).getTimeStamp()+"," +t +","+ model.getClasses().get(i).getLabel());
            }
        }
        CE_info.close();
    }

    private void updateUnknownRatio(Model model) {
        float sumUnkRM = 0;
        for (int i = 0; i < model.getShortTimeMemory().getData().size(); i++) {
            Set<String> labels = DataSetUtils.getLabelSet(model.getShortTimeMemory().getData().get(i));
            for (Iterator<String> iterator = labels.iterator(); iterator.hasNext();) {
                String next = iterator.next();
                super.getConsufionMtx().get(next)[0][2] += 1;
            }
        }
        
        
        for (Map.Entry<String, int[][]> entry : getConsufionMtx().entrySet()) {
                int[][] value = entry.getValue();
                float unkRM = ((float) value[0][2] / (float)(value[0][0] + value[0][1] +  value[0][2])); //unknwon ration
                if(Float.isNaN(unkRM)){
                    unkRM = 0;
                }
                sumUnkRM += unkRM;
            }
        float auxUnkRM = (float)(sumUnkRM / (float)this.getConsufionMtx().size());
        if(Float.isNaN(auxUnkRM))
            auxUnkRM = 0;
        this.getUnkRM().add(auxUnkRM);
    }
    
    /**
     * @return the qtdeNP
     */
    public int getQtdeNP() {
        return qtdeNP;
    }

    /**
     * @param qtdeNP the qtdeNP to set
     */
    public void setQtdeNP(int qtdeNP) {
        this.qtdeNP = qtdeNP;
    }

//    /**
//     * @return the deletedExamples
//     */
//    public ArrayList<Integer> getDeletedExamples() {
//        return deletedExamples;
//    }
}
