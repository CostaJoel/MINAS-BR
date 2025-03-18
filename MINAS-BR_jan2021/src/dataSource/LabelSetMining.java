package dataSource;


/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import com.yahoo.labs.samoa.instances.Instance;
import static dataSource.DataSetUtils.getLabelSet;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import static java.util.Map.Entry.comparingByValue;
import java.util.Set;
import static java.util.stream.Collectors.toMap;
import moa.streams.MultiTargetArffFileStream;
import weka.core.Instances;

/**
 *
 * @author Joel
 */
public class LabelSetMining {
    
//    public static void main(String[] args) throws Exception {
//////        String dataSetPath = "D:\\datasets\\reais\\mediamill\\mediamill_modified.arff";
//////        String dataSetPath = "D:\\datasets\\sinteticos\\MOA\\ConceptEvolution\\MOA-5C-7C-2D-0.16R\\MOA-5C-7C-2D-0.16R.arff";
//////        String dataSetPath = "D:\\datasets\\sinteticos\\vinicius\\4CRE-V2-CE-ML.arff";
//        String dataSetPath = "D:\\datasets\\reais\\reuters\\reuters_norm_modified.arff";
//        String dataSetName = "reuters";
//        int L =40;
//////        removeNoise(dataSetPath, "reuters");
//////        String dataSetPath = "D:\\datasets\\reais\\yeast\\yeast-V2_modified.arff";
//////        String dataSetPath = "D:\\datasets\\reais\\scene\\scene.arff";
//////        String dataSetPath = "D:\\datasets\\reais\\nus-wide-128D\\nus-wide-full-cVLADplus-test.arff";
////        String train = "D:\\datasets\\reais\\nus-wide-128D\\nus-wide-train.arff";
////        String test = "D:\\datasets\\reais\\nus-wide-128D\\nus-wide-test.arff";
////        String dataSetName = "nus-wide";
////        int L = 81;
//////        String dataSetName = "scene";
//////        String dataSetName = "mediamill";
//////        String dataSetName = "reuters";
////        String newDataSetPath = removeInfrequentLabels(dataSetPath, dataSetName, L, 0.01);
////        String newDataSetPath = removeInfrequentLabels(dataSetPath, dataSetName, L, 0.05);
////        removeNoise(dataSetPath, dataSetName, L);
//////        Instances D_ = DataSetUtils.dataFileToInstance(dataSetPath);
////        
//        Set<String> novelties = LabelSetMining.getNovelties(dataSetPath, L, 0.1, 0.9);
////
//        ArrayList<Instance> D = new ArrayList<Instance>();
//        ArrayList<Instance> train = new ArrayList<Instance>();
//        ArrayList<Instance> test = new ArrayList<Instance>();
////        
//        MultiTargetArffFileStream file = new MultiTargetArffFileStream(dataSetPath, String.valueOf(L));
//        file.prepareForUse();
//        while(file.hasMoreInstances()){
//            D.add(file.nextInstance().getData());
//        }
////        file.restart();
//////        Instance D_ = removeNoise(dataSetPath, "reuters");
////        double x = DataSetUtils.getCardinality(D, L);
////        train = D.subList(0, (int)(D.size()*0.1));
////        test = D.subList(train.size(), D.size());
//////        file.restart();
//////        int[] dist = DataSetUtils.getLabelsDistribution(train);
////            DataSetUtils.slipTrainTest(train, test, file, D.size(), 0.1);
////            int[] dist = DataSetUtils.getLabelsDistribution(train);
//        DataSetUtils.noveltyDetectionStratificationNusWide(D, train, test, novelties, 25, 0.1);
//        int[] dist = DataSetUtils.getLabelsDistribution(train);
//        DataSetUtils.createFileBasedOnList(train, dataSetPath, "train", dataSetName);
//        DataSetUtils.createFileBasedOnList(test, dataSetPath, "test", dataSetName);
//    }
    
    public static double[] getMeanValues(ArrayList<Instance> D){
        double[] mean = new double[D.get(0).numInputAttributes()];
        for (Instance inst : D) {
            for (int i = 0; i < inst.numInputAttributes(); i++) {
                mean[i] += inst.valueInputAttribute(i);
            }
        }
        for (int i = 0; i < mean.length; i++) {
            double aux = mean[i]/D.size();
            mean[i] = aux;
        }
        
        return mean;
    }
    
    public static void replaceMissingValues(ArrayList<Instance> D, double[] mean){
        for (Instance inst : D) {
//             System.out.println(Arrays.toString(inst.toDoubleArray()));
            for (int i = 0; i < inst.numInputAttributes(); i++) {
                if(inst.valueInputAttribute(i) == 0){
                    inst.setValue(inst.inputAttribute(i), mean[i]);
                }
            }
//            System.out.println(Arrays.toString(inst.toDoubleArray()));
        }
    }
    
    
    /**
     * Get the frequent label sets from the examples and creates a new dataset
     * @param dataSetPath
     * @param support minimum label set support
     * @return
     * @throws Exception 
     */
    
    /**
     * Creates a new file representing the new pre-processed dataset
     * @param D
     * @param m
     * @param frequentLabelSets
     * @param dataSetPath
     * @param dataSetName
     * @throws IOException 
     */
    private static void transformDataSet(ArrayList<Instance> D, int m, ArrayList<Set<String>> frequentLabelSets, String dataSetPath, String dataSetName) throws IOException{
        Set<String> L = new HashSet<>();
        for (Set<String> key : frequentLabelSets) {
            L.addAll(key);
        }

        String fileName = dataSetPath.replace(".arff", "_modified_5.arff");
        FileWriter file = new FileWriter(new File(fileName), false);
        file.write("@relation '" + dataSetName + ": -C "+ (L.size()-1) + "'\n");
        file.write("\n");
        
        for (int i = 0; i < m; i++) {
            if(L.contains(""+i) && i != 67){
                file.write("@attribute class"+i+" {0, 1}\n");
            }
        }

        int numAtt = D.get(0).numInputAttributes();
        
        for (int i = 0; i < numAtt; i++) {
            file.write("@attribute " + D.get(0).inputAttribute(i).name() + " numeric\n");
        }
        file.write("\n");
        file.write("@data\n");
        
        for (int i = 0; i < D.size(); i++) {
            for (int j = 0; j < m; j++) {
                if(L.contains(""+j) && j != 67){
                    file.write(((int)D.get(i).value(j))+",");
                }
            }
            for (int j = 0; j < numAtt; j++) {
                if(j < numAtt-1)
                    file.write(D.get(i).valueInputAttribute(j)+",");
                else
                    file.write(D.get(i).valueInputAttribute(j)+"\n");
            }
        }
        file.close();
    }
    /**
     * Creates a new file representing the new pre-processed dataset
     * @param D
     * @param m
     * @param frequentLabelSets
     * @param dataSetPath
     * @param dataSetName
     * @throws IOException 
     */
    private static void MaximalItemSetsTransformationDataset(ArrayList<Instance> D, ArrayList<Set<String>> MIs, String dataSetPath, String dataSetName) throws IOException{
        String fileName = dataSetPath.replace(".arff", "_maximalItemsets_v2.arff");
        FileWriter file = new FileWriter(new File(fileName), false);
        file.write("@relation '" + dataSetName + "-maximalItemsets: -C "+ (MIs.size()) + "'\n");
        file.write("\n");
        
        for (Iterator<Set<String>> iterator = MIs.iterator(); iterator.hasNext();) {
            Set<String> next = iterator.next();
            System.out.println(next);
            file.write("@attribute class"+next.toString().replaceAll(", ", "-")+" {0, 1}\n");
        }
        
        int numAtt = D.get(0).numInputAttributes();
        for (int i = 0; i < numAtt; i++) {
            file.write("@attribute " + D.get(0).inputAttribute(i).name() + " numeric\n");
        }
        
        file.write("\n");
        file.write("@data\n");
        
        for (int i = 0; i < D.size(); i++) {
            boolean remove = true;
            String aux = "";
            for (int j = 0; j < MIs.size(); j++) {
                Set<String> Y = DataSetUtils.getLabelSet(D.get(i));
                Y.retainAll(MIs.get(j));
                if(Y.containsAll(MIs.get(j))){
                    aux = aux+"1,";
                    remove = false;
                }else{
                    aux = aux+"0,";
                }
            }
            if(!remove){
                file.write(aux);
                for (int j = 0; j < numAtt; j++) {
                    if(j < numAtt-1)
                        file.write(D.get(i).valueInputAttribute(j)+",");
                    else
                        file.write(D.get(i).valueInputAttribute(j)+"\n");
                }
            }
        }
        file.close();
    }
    
    /**
     * Removes examples which do not assigned to frequent labels
     * @param D
     * @param frequentLabelSets
     * @return 
     */
    public static ArrayList<Instance> removeInfrequentLabels(ArrayList<Instance> D, ArrayList<Set<String>> frequentLabelSets){
        Set<String> L = new HashSet<>();
        ArrayList<Instance> D_reduced= new ArrayList<Instance>();
        for (Set<String> key : frequentLabelSets) {
            L.addAll(key);
        }

        for (int i=0; i < D.size(); i++){
            Set<String> Y = getLabelSet(D.get(i));
            Y.retainAll(L);
            if(!Y.isEmpty()){
                D_reduced.add(D.get(i));
            }else{
                System.out.println("Example: "+ i + " - Infrequent Labelset: " + getLabelSet(D.get(i)).toString());
            }
        }
        return D_reduced;
    }
    
    /**
     * 
     * @param dataSetPath original data set path
     * @param dataSetName original data set name
     * @param minSupport
     * @return the new data set path with infrequent labels removed
     * @throws Exception 
     */
    public static String removeInfrequentLabels(String dataSetPath, String dataSetName, int m, double minSupport) throws Exception {
        Instances D_ = DataSetUtils.dataFileToInstance(dataSetPath);
        weka.core.Instance inst = D_.instance(0);
//        int m = D_.classIndex();
        
        MultiTargetArffFileStream file = new MultiTargetArffFileStream(dataSetPath, String.valueOf(m));
//        file.getHeader().deleteAttributeAt(m);
        file.prepareForUse();
        
        
        ArrayList<Instance> D = new ArrayList<Instance>();
        while(file.hasMoreInstances()){
            D.add(file.nextInstance().getData());
        }
        
        int[] dist = DataSetUtils.getLabelsDistribution(D);
        int t = (int) (D.size()*minSupport);
        Set<String> frequentLabels = new HashSet<>();
        for (int i = 0; i < dist.length; i++) {
//            System.out.println(D.get(0).outputAttribute(i).name());
            if(dist[i] > t){
                System.out.println(D.get(0).outputAttribute(i).name());
                frequentLabels.add(""+i);
            }
        }
        
        return createNewFile(D, frequentLabels, dataSetPath, dataSetName);
    }
    /**
     * 
     * @param dataSetName original data set name
     * @param minSupport
     * @return the new data set path with infrequent labels removed
     * @throws Exception 
     */
    public static String removeInfrequentLabels(String dataSetPathTrain, String dataSetPathTest, String dataSetName, int m, double minSupport) throws Exception {
        MultiTargetArffFileStream file = new MultiTargetArffFileStream(dataSetPathTrain, String.valueOf(m));
        file.prepareForUse();
        ArrayList<Instance> D = new ArrayList<Instance>();
        while(file.hasMoreInstances()){
            Instance ist = file.nextInstance().getData();
//            System.out.println(ist.attribute(m).name());
//            System.out.println(ist.value(m));
            ist.deleteAttributeAt(m);
//            System.out.println(ist.value(m));
            D.add(ist);
        }
        file = new MultiTargetArffFileStream(dataSetPathTest, String.valueOf(m));
        file.prepareForUse();
        while(file.hasMoreInstances()){
            Instance ist = file.nextInstance().getData();
//            System.out.println(ist.attribute(m).name());
            ist.deleteAttributeAt(m);
            D.add(ist);
        }
        
        int[] dist = DataSetUtils.getLabelsDistribution(D);
        int t = (int) (D.size()*minSupport);
        Set<String> frequentLabels = new HashSet<>();
        for (int i = 0; i < dist.length; i++) {
//            System.out.println(D.get(0).outputAttribute(i).name());
            if(dist[i] > t){
                System.out.println(D.get(0).outputAttribute(i).name());
                frequentLabels.add(""+i);
            }
        }
        
        return createNewFile(D, frequentLabels, dataSetPathTrain, dataSetName);
    }

    private static String createNewFile(ArrayList<Instance> D, Set<String> frequentLabels, String dataSetPath, String dataSetName) throws IOException {
        String fileName = dataSetPath.replace(".arff", "_modified.arff");
        FileWriter file = new FileWriter(new File(fileName), false);
        file.write("@relation '" + dataSetName + "_FL: -C "+ (frequentLabels.size()) + "'\n");
        file.write("\n");
        
        for (int i = 0; i < D.get(0).numOutputAttributes(); i++) {
            if(frequentLabels.contains(""+i)){
                file.write("@attribute "+D.get(0).outputAttribute(i).name()+" {0, 1}\n");
            }
        }
//        for (Iterator<String> iterator = frequentLabels.iterator(); iterator.hasNext();) {
//            String next = iterator.next();
//            file.write("@attribute "+next+" {0, 1}\n");
//        }
        
        int numAtt = D.get(0).numInputAttributes();
        for (int i = 1; i < numAtt; i++) {
            file.write("@attribute " + D.get(0).inputAttribute(i).name() + " numeric\n");
        }
        
        file.write("\n");
        file.write("@data\n");
        
        for (Instance instance : D) {
            String aux = "";
            boolean remove = true;
            Set<String> Y = DataSetUtils.getLabelSet(instance);
            Y.retainAll(frequentLabels);
            if(!Y.isEmpty()){
                Set<String> aux2 = DataSetUtils.getLabelSet(instance);
                System.out.println(aux2);
                if(!aux2.isEmpty()){
                    for (int i = 0; i < instance.numOutputAttributes(); i++) {
                        if(frequentLabels.contains(""+i)){
                            aux = aux+((int)instance.valueOutputAttribute(i))+",";
                            remove = false;
                            aux2.add(instance.outputAttribute(i).name());
                        }
                    }
                    if(!remove){
                        file.write(aux);
                        System.out.println(aux);
                        for (int i = 0; i < instance.numInputAttributes()-1; i++) {
                            if(i < instance.numInputAttributes()-2)
                                file.write(instance.valueInputAttribute(i)+",");
                            else
                                file.write(instance.valueInputAttribute(i)+"\n");
                        }
                    }
                }
            }
        }
        file.close();
        return fileName;
    }
    
    
    /**
     * 
     * @param newDataSetPath data set path
     * @param m number of labels
     * @param tHigh threshold for high dependency degree labels
     * @param tLow threshold for low dependency degree labels
     * @return the novelty class labels index
     * @throws IOException 
     */
    public static Set<String> getNovelties(String newDataSetPath, int m, double tHigh, double tLow) throws IOException {
        MultiTargetArffFileStream newFile = new MultiTargetArffFileStream(newDataSetPath, String.valueOf(m));
        newFile.prepareForUse();
        
        ArrayList<Instance> D = new ArrayList<Instance>();
        while(newFile.hasMoreInstances()){
            D.add(newFile.nextInstance().getData());
        }
        
        ArrayList<String> auxRemove = new ArrayList<>();
        
        HashMap<String, Float> dependenceDegrees = DataSetUtils.getDependencesDegree(D);
        for (Map.Entry<String, Float> entry : dependenceDegrees.entrySet()) {
            String key = entry.getKey();
            Float value = entry.getValue();
            if(value == 0){
                auxRemove.add(key);
            }
        }
        
        for (String key : auxRemove) {
            dependenceDegrees.remove(key);
        }
//        System.out.println(dependenceDegrees.get("4,5"));
//        System.out.println(dependenceDegrees.get("5,4"));
//        System.out.println(dependenceDegrees.get("4,3"));
//        System.out.println(dependenceDegrees.get("3,4"));
        HashMap<String, Float> sorted = dependenceDegrees.entrySet().stream().sorted(Collections.reverseOrder(Map.Entry.comparingByValue())).
                collect(toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e2, LinkedHashMap::new));
        Set<String> L_removed = new HashSet<>();
        Set<String> L_novelty = new HashSet<>();
        
        int contHigh = (int) ((int) sorted.size()*tHigh);
        int contLow = (int) ((int) sorted.size()*tLow);
        int cont = 0;
        for (Map.Entry<String, Float> entry : sorted.entrySet()) {
            String key = entry.getKey();
//            Float value = entry.getValue();
            if(cont < contHigh){
                String aux[] = key.split(",");
                L_removed.add(aux[0]);
                L_removed.add(aux[1]);
            }
            
            if(cont > contLow){
                String aux[] = key.split(",");
                L_novelty.add(aux[0]);
                L_novelty.add(aux[1]);
            }
            cont++;
        }
        System.out.println("**********get novelties***********");
        System.out.println(L_removed);
        System.out.println(L_novelty);
        L_novelty.removeAll(L_removed);
        System.out.println(L_novelty);
        return L_novelty;
    }

    private static HashMap<String, Float> sortByDependenceDegrees(HashMap<String, Float> dependenceDegrees) {
        System.out.println("map after sorting by values in descending order: "
                + dependenceDegrees);
        HashMap<String, Float> sorted = dependenceDegrees.entrySet().stream().sorted(Collections.reverseOrder(Map.Entry.comparingByValue())).
                collect(toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e2, LinkedHashMap::new));

        System.out.println("map after sorting by values in descending order: "
                + sorted);
        return sorted;
    }

    private static Set<String> getMergedLabelSets(HashMap<String, Float> sortedDependenceDegrees, int m, float t1, float t2) {
        Set<String> L = new HashSet<>();
        Set<String> L_merged = new HashSet<>();
        Set<String> L_removed = new HashSet<>();
        Set<String> L_novelty = new HashSet<>();
        for (int i = 0; i < m; i++) {
            L.add(""+i);
        }
        
        for (Map.Entry<String, Float> entry : sortedDependenceDegrees.entrySet()) {
            String key = entry.getKey();
            Float value = entry.getValue();
            if(value>t1){
                String aux[] = key.split(",");
                L_removed.add(aux[0]);
                L_removed.add(aux[0]);
            }else if(value < t2){
                String aux[] = key.split(",");
                L_novelty.add(aux[0]);
                L_novelty.add(aux[1]);
            }
        }
        L.removeAll(L_removed);
        L.addAll(L_merged);
        return L;
    }

    private static void createFileWithMergedLabels(Set<String> L_new, ArrayList<Instance> D, String dataSetPath, String dataSetName) throws IOException {
        String fileName = dataSetPath.replace(".arff", "_merged.arff");
        FileWriter file = new FileWriter(new File(fileName), false);
        file.write("@relation '" + dataSetName + "_FL_merged: -C "+ (L_new.size()) + "'\n");
        file.write("\n");
        
        Set<String> aux = new HashSet<>();
        aux.addAll(L_new);
        HashMap<String, Integer> orderLabels = new HashMap<>(); //Store the order of labels to create a bitation vector 
        int cont = 0;
        
        //Remaining labels
        for (int i = 0; i < D.get(0).numOutputAttributes(); i++) {
            if(aux.contains(""+i)){
                orderLabels.put(""+i, cont);
                cont++;
                aux.remove(""+i);
//                System.out.println(""+i);
//                System.out.println("@attribute "+D.get(0).outputAttribute(i).name()+" {0, 1}");
                file.write("@attribute "+D.get(0).outputAttribute(i).name()+" {0, 1}\n");
            }
        }
        
        //New merged labels
        for (Iterator<String> iterator = aux.iterator(); iterator.hasNext();) {
            String next = iterator.next();
            orderLabels.put(next, cont);
            cont++;
            String[] aux2 = next.split(",");
//            System.out.println(Arrays.toString(aux2));
            file.write("@attribute ");
            for (int i = 0; i < aux2.length; i++) {
//                System.out.print(D.get(0).outputAttribute(Integer.parseInt(aux2[i])).name() + "_");
                file.write(D.get(0).outputAttribute(Integer.parseInt(aux2[i])).name() + "_");
            }
            file.write(" {0, 1}\n");
        }
        
        int numAtt = D.get(0).numInputAttributes();
        for (int i = 0; i < numAtt; i++) {
            file.write("@attribute " + D.get(0).inputAttribute(i).name() + " numeric\n");
        }
        file.write("\n");
        file.write("@data\n");
        String[] positions = new String[L_new.size()];
        for (Instance instance : D) {
            Set<String> Y = DataSetUtils.getLabelSet(instance);
            if(!Y.isEmpty()){
                //fill the output attributes take into account the new labels merged
                boolean noise = true;
                for (Iterator<String> iterator = L_new.iterator(); iterator.hasNext();) {
                    String next = iterator.next();
                    String[] aux2 = next.split(",");
                    boolean flag = true; 
                    for (int i = 0; i < aux2.length; i++) {
                        if(!Y.contains(aux2[i])){
                            flag = false;
                            break;
                        }
                    }
                    if(flag){
                        noise = false;
                        positions[orderLabels.get(next)] = "1";
                    }else{
                        positions[orderLabels.get(next)] = "0";
                    }
                }
                if(!noise){
                    String aux2 = Arrays.toString(positions).replace("[", "").replace("]", "");
                    file.write(aux2.trim()+",");
        //            System.out.print(aux2.trim()+",");

                    //fill the input attribues
                    for (int i = 0; i < instance.numInputAttributes(); i++) {
                        if(i < instance.numInputAttributes()-1){
                            file.write(instance.valueInputAttribute(i)+",");
        //                    System.out.print(instance.valueInputAttribute(i)+",");
                        }else{
                            file.write(instance.valueInputAttribute(i)+"\n");
        //                    System.out.print(instance.valueInputAttribute(i)+"\n");
                        }
                    }
                }
            }
//                }else{
//                    System.out.println(Arrays.toString(instance.toDoubleArray()));
//                }
//            }else{
//                System.out.println(Arrays.toString(instance.toDoubleArray()));
//            }
        }
        file.close();
    }

    public static void removeNoise(String dataSetPath, String dataSetName, int m) throws Exception {
        MultiTargetArffFileStream file = new MultiTargetArffFileStream(dataSetPath, String.valueOf(m));
        file.prepareForUse();
        
        ArrayList<Instance> D = new ArrayList<Instance>();
        
        while(file.hasMoreInstances()){
            Instance instance = file.nextInstance().getData();
            Set<String> Y = DataSetUtils.getLabelSet(instance);
            if(!Y.isEmpty()){
                D.add(instance);
            }else{
                System.out.println(Arrays.toString(instance.toDoubleArray()));
            }
        }
        
        FileWriter fileWriter = new FileWriter(new File(dataSetPath), false);
        fileWriter.write("@relation '" + dataSetName + ": -C "+ m + "'\n");
        fileWriter.write("\n");
        
        for (int i = 0; i < D.get(0).numOutputAttributes(); i++) {
            fileWriter.write("@attribute "+D.get(0).outputAttribute(i).name()+" {0, 1}\n");
        }
//        for (Iterator<String> iterator = frequentLabels.iterator(); iterator.hasNext();) {
//            String next = iterator.next();
//            file.write("@attribute "+next+" {0, 1}\n");
//        }
        
        int numAtt = D.get(0).numInputAttributes();
        for (int i = 0; i < numAtt; i++) {
            System.out.println("@attribute " + D.get(0).inputAttribute(i).name() + " numeric");
            fileWriter.write("@attribute " + D.get(0).inputAttribute(i).name() + " numeric\n");
        }
        
        fileWriter.write("\n");
        fileWriter.write("@data\n");
        
        for (int i = 0; i < D.size(); i++) {
            String labels = "";
            for (int j = 0; j < D.get(i).numOutputAttributes(); j++) {
                labels = labels+(int)D.get(i).valueOutputAttribute(j) + ",";
            }
            fileWriter.write(labels);
            for (int j = 0; j < D.get(i).numInputAttributes(); j++) {
                if(j < D.get(i).numInputAttributes()-1){
                    fileWriter.write(D.get(i).valueInputAttribute(j)+",");
//                    System.out.print(instance.valueInputAttribute(i)+",");
                }else{
                    fileWriter.write(D.get(i).valueInputAttribute(j)+"\n");
//                    System.out.print(instance.valueInputAttribute(i)+"\n");
                }
            }
        }
        fileWriter.close();
    }
}
