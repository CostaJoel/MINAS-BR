package com.minasbr.dataSource;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Comparator;
import java.util.TreeMap;
import meka.core.MLUtils;
import moa.classifiers.lazy.kNN;
import moa.core.InstanceExample;
import moa.streams.MultiTargetArffFileStream;
import weka.core.AbstractInstance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author gerson
 */
public class DataSetUtils {

    /**
     * loadDataset
     *
     * @param filename
     * @return	the dataset
     * @throws java.lang.Exception
     */
    public static Instances loadDataset(String filename) throws Exception {

        Instances D = null;

        // Check for filename
        if (filename == null || filename.isEmpty()) {
            throw new Exception("[Error] You did not specify a dataset!");
        }

        // Check for existence of file
        File file = new File(filename);
        if (!file.exists()) {
            throw new Exception("[Error] File does not exist: " + filename);
        }
        if (file.isDirectory()) {
            throw new Exception("[Error] " + filename + " points to a directory!");
        }

        try {
            DataSource source = new DataSource(filename);
            D = source.getDataSet();
        } catch (Exception e) {
            e.printStackTrace();
            throw new Exception("[Error] Failed to load Instances from file '" + filename + "'.");
        }

        return D;
    }
    
      public static Set<String> getClassesConhecidas(Instances train, int L) {
        int[] dist = DataSetUtils.getLabelsDistribution(train, L);
        Set<String> classesConhecidas = new HashSet<>();
        for (int j = 0; j < train.size(); j++) {
            for (int k = 0; k < L; k++) {
                if(train.get(j).value(k) == 1 && dist[k] > 10){
                    classesConhecidas.add(String.valueOf(k));
                }
            }
        }
        return classesConhecidas;
    }      
      
      public static int[] getLabelsDistribution(Instances D, int L){
        int[] dist = new int[L];
        for (int i = 0; i < D.size(); i++) {
            for (int j = 0; j < L; j++) {
                if(D.get(i).value(j) == 1){
                    dist[j]++;
                }
            }
        }
        return dist;
    }
      
      /**
     * Convert a prediction matrix (meka prediction) to prediction set 
     * @param trueLabels
     * @return 
     */
    public static ArrayList<Set<String>> predictionMatrixToSet(int[][] trueLabels) {
        ArrayList<Set<String>> pred = new ArrayList<Set<String>>();
        for (int i = 0; i < trueLabels.length; i++) {
            HashSet<String> set = new HashSet<String>();
            for (int j = 0; j < trueLabels[i].length; j++) {
                if(trueLabels[i][j] > 0)
                    set.add(""+j);
            }
            pred.add(set);
        }
        return pred;
    }
    
    public static int getNumLabels(String filePath) throws Exception{
        Instances dataSetAux = DataSetUtils.loadDataset(filePath);
        MLUtils.prepareData(dataSetAux);
        return dataSetAux.classIndex();
    }
    
    
    
    public static String mergeArffFiles(ArrayList<String> filePaths, ArrayList<Integer> Ls, int numAtt) throws IOException{
        int Lall = 0;
        String fileName = filePaths.get(0).replace(".arff", "_final.arff");
        FileWriter finalFile = new FileWriter(new File(fileName), false);
        for (Integer L : Ls) {
            if(L > Lall)
                Lall = L;
        }
        finalFile.write("@relation 'MOA: -C "+ Lall+ "'\n");
        finalFile.write("\n");
        for (int i = 0; i < Lall; i++) {
            finalFile.write("@attribute class"+i+" {0, 1}\n");
        }
        for (int i = 0; i < numAtt; i++) {
            finalFile.write("@attribute att"+i+" numeric\n");
        }
        finalFile.write("\n");
        finalFile.write("@data\n");
        
        for (int i = 0; i < filePaths.size(); i++) {
            MultiTargetArffFileStream stream = new MultiTargetArffFileStream(filePaths.get(i), String.valueOf(Ls.get(i)));
            stream.prepareForUse();
            int c = 0;
            while(stream.hasMoreInstances()){
                c++;
                System.out.println(c);
                Instance inst = stream.nextInstance().instance;
                int L = Ls.get(i);
                if(L < Lall){
                    for (int j = 0; j < L; j++) {
                        int value = (int) inst.valueOutputAttribute(j);
                        finalFile.write(value+",");
                    }
                    for (int j = 0; j < (Lall - L); j++) {
                        finalFile.write("0,");
                    }
                }else{
                    for (int j = 0; j < L; j++) {
                        int value = (int) inst.valueOutputAttribute(j);
                        finalFile.write(value+",");
                    }
                }
                
                for (int j = 0; j < inst.numInputAttributes(); j++) {
//                    System.out.println(inst.valueInputAttribute(i));
                    if(j ==  (inst.numInputAttributes()-1))
                        finalFile.write(inst.valueInputAttribute(i)+"\n");
                    else
                        finalFile.write(inst.valueInputAttribute(i)+",");
                }
            }
        }
        finalFile.close();
        return fileName;
    }
    
    /**
     * Return cardinalities of all windows
     * @param dataSet 
     * @param L
     * @return 
     */
    public static float getCardinality(List<Instance> dataSet, int L){
        float sum = 0;
        for (int i = 0; i < dataSet.size(); i++) {
            Set<String> labels = DataSetUtils.getLabelSet(dataSet.get(i));
           sum += labels.size();
        }
        float z = (float)sum / (float) dataSet.size();
        return z;
    }
    
    /**
     * Return cardinalities of all windows
     * @param dataSet 
     * @param w 
     * @param L
     * @return 
     */
    public static float[] getWindowsCardinalities(ArrayList<Instance> dataSet, int w, int L){
        float[] cardinalities = new float[dataSet.size()/w];
        for (int i = 0; i < cardinalities.length; i++) {
            cardinalities[i] = getCardinality(dataSet.subList(i*w, (i*w)+w), L);
        }
        return cardinalities;
    }
    
    /**
     * Return instance labels
     * @param x - AbstractInstance
     * @param L number of labels
     * @return labels - int[]
     */
    public static int[] getBipartition(Instance x, int L){
        int[] Y = new int[L];
        double[] aux = x.toDoubleArray();
        for (int i = 0; i < L; i++) {
            int teste = x.classIndex();
            if(aux[x.classIndex()+i] > 0){
                Y[x.classIndex()+i] = 1;
            }
        }
        return Y;
    }
    
    public static <T> ArrayList<T> removeItensReptidos(ArrayList<T> valores){
        ArrayList<T> novaLista = new ArrayList<>();
        T ultimo = null;
        boolean repetido = false;
        for (T j : valores) {

            if (j.equals(ultimo)) {
                repetido = true;
            } else {

                if (repetido) {
                    novaLista.add(ultimo);
                }
                novaLista.add(j);
                repetido = false;

            }
            ultimo = j;
        }

        if (repetido) {
            novaLista.add(ultimo);
        }
        return novaLista;
    }
    
    /**
     * Return instance labels
     * @param x - AbstractInstance
     * @return labels - String
     */
    public static String getLabels(AbstractInstance x){
        double[] labelsAux = x.toDoubleArray();
        
        ArrayList<String> aux = new ArrayList<>();
        for (int i = 0; i < x.classIndex(); i++) {
            if(labelsAux[i] > 0){
                aux.add(String.valueOf(i));
            }
        }
        //System.out.println(aux.toString());
        return aux.toString();
    }
    
    /**
     * Retorna os rótulos dos exemplos
     * @param x exemplo
     * @param nLabel quantidade total de classes do problema
     * @return 
     */
    public static ArrayList<String> getLabels(Instance x){
        double[] labelsAux = x.toDoubleArray();
        
        ArrayList<String> aux = new ArrayList<>();
        for (int i = 0; i < x.numOutputAttributes(); i++) {
            if(labelsAux[i] > 0){
                aux.add(""+i);
            }
        }
        //System.out.println(aux.toString());
        return aux;
    }
    /**
     * Retorna os rótulos dos exemplos
     * @param x exemplo
     * @param nLabel quantidade total de classes do problema
     * @return 
     */
    public static Set<String> getLabelSet(Instance x){
        double[] labelsAux = x.toDoubleArray();
        
        Set<String> aux = new HashSet<>();
        for (int i = 0; i < x.numOutputAttributes(); i++) {
            if(labelsAux[i] > 0){
                aux.add(""+i);
            }
        }
        //System.out.println(aux.toString());
        return aux;
    }
    
    /**
     * Retorna os rótulos dos exemplos
     * @param x exemplo
     * @param nLabel quantidade total de classes do problema
     * @return 
     */
    public static Set<String> getLabelSetNames(Instance x){
        double[] labelsAux = x.toDoubleArray();
        
        Set<String> aux = new HashSet<>();
        for (int i = 0; i < x.numOutputAttributes(); i++) {
            if(x.valueOutputAttribute(i)>0){
                aux.add(x.outputAttribute(i).name());
            }
        }
        //System.out.println(aux.toString());
        return aux;
    }
    
    /**
     * Retorna os rótulos dos exemplos
     * @param x exemplo
     * @return 
     */
    public static int[] getItemset(Instance x){
        double[] labelsAux = x.toDoubleArray();
        int[] aux = new int[x.numOutputAttributes()];
        int index = 0;
        for (int i = 0; i < x.numOutputAttributes(); i++) {
            if(labelsAux[i] > 0){
                aux[index] = i;
                index++;
            }
        }
        //System.out.println(aux.toString());
        return aux;
    }
    
    /**
     * Retorna a quantidade de exemplos de cada classe
     * @param D base de dados
     * @param nLabel quantidade de rótulos
     * @return 
     */
    public static int[] getLabelsDistribution(List<Instance> D){
        int[] dist = new int[D.get(0).numOutputAttributes()];
        for (int i = 0; i < D.size(); i++) {
            for (int j = 0; j < D.get(0).numOutputAttributes(); j++) {
                if(D.get(i).valueOutputAttribute(j) == 1){
                    dist[j]++;
                }
            }
        }
        return dist;
    }
    
    public static int[] getLabelsDistribution(List<InstanceExample> D, int L){
        int[] dist = new int[L];
        for (int i = 0; i < D.size(); i++) {
            for (int j = 0; j < L; j++) {
                if(D.get(i).getData().value(j) == 1){
                    dist[j]++;
                }
            }
        }
        return dist;
    }
    
    /**
     * Transform a file's line in a instance
     * @param line
     * @return 
     */
    public static Instance getValueFile(String line){
        String[] aux = line.split(",");
        double[] auxD = new double[aux.length];
        for (int i = 0; i < aux.length; i++) {
            auxD[i] = Double.parseDouble(aux[i].trim());
        }
        return new DenseInstance(1, auxD);
    }
    
    /**
     * Return a HashMap(K,V), where K represents the class and V it is a list containing the examples of each class. 
     * @param D training examples
     * @param L number of labels
     * @return
     * @throws Exception 
     */
    public static HashMap<String, ArrayList<Instance>> separateBR(ArrayList<Instance> D, int L) throws Exception{
        //K -> Classe
        //V -> Lista de exemplos de cada classe
        HashMap<String, ArrayList<Instance>> BRSubSet = new HashMap();
        
        for (int i = 0; i < L; i++) { //Para cada classe uma lista é criada. Essa lista irá conter os exemplos de cada classe
            ArrayList<Instance> lista = new ArrayList<>();
            BRSubSet.put(""+i, lista);
        }
        
        for (int i = 0; i < D.size(); i++) {
            ArrayList<String> labels = getLabels(D.get(i)); //Pega os rótulos da instancia
            ArrayList<Instance> generic; //Lista auxilar usada para receber a lista do hashmap, o exemplo é adicionado a está lista e depois retorna para a HashMap
            for (String label : labels) { //Para cada classe do exemplo
                generic = BRSubSet.get(label); //busca a lista referente a classe "label"
                generic.add((Instance) D.get(i)); //Adiciona o exemplo à lista
                BRSubSet.put(label, generic); //Retorna a lista para a classe
            }
        }
        
        return BRSubSet;
    }
    
    /**
     * Separa os dados rotulados que serão usados na fase offline para criação do modelo
     * @param D base de dados completa
     * @param percent porcentagem de exemplos que será usado para treino
     * @return base de dados de treino e base de dados que será usada no fluxo contínuo de dados
     */
    public static ArrayList<ArrayList<Instance>> separateTrain(Instances D, float percent){
        int i = 0;
        ArrayList<ArrayList<Instance>> D_ = new ArrayList<>();
        ArrayList<Instance> train = new ArrayList<>();
        ArrayList<Instance> fcd = new ArrayList<>();
        Instance inst;
        while( i < D.numInstances()*percent ) {
            inst = new DenseInstance(1, D.get(i).toDoubleArray());
            train.add(inst);
            i++;
        }
        for (int j = i; j < D.numInstances(); j++) {
            inst = new DenseInstance(1, D.get(j).toDoubleArray());
            fcd.add(inst);
        }
        D_.add(train);
        D_.add(fcd);
        return D_;
    }
    
    /**
     * Separa os dados rotulados que serão usados na fase offline para criação do modelo
     * @param D base de dados completa
     * @param percent porcentagem de exemplos que será usado para treino
     * @return base de dados de treino e base de dados que será usada no fluxo contínuo de dados
     */
    public static ArrayList<ArrayList<Instance>> separateTrain(ArrayList<Instance> D, float percent){
        int i = 0;
        ArrayList<ArrayList<Instance>> D_ = new ArrayList<>();
        ArrayList<Instance> train = new ArrayList<>();
        ArrayList<Instance> fcd = new ArrayList<>();
        while( i < D.size()*percent ) {
            train.add(D.get(i));
            i++;
        }
        for (int j = i; j < D.size(); j++) {
            fcd.add(D.get(j));
        }
        D_.add(train);
        D_.add(fcd);
        return D_;
    }
    
    /**
     * Lê um arquivo e tranforma os exemplos em um Array de exemplos
     * @param fileName caminho do arquivo
     * @return array com os exemplos em formato de Instance
     * @throws Exception 
     */
    public static ArrayList<Instance> dataFileToArray(String fileName) throws Exception {
        Instances D = dataFileToInstance(fileName);
        int i = 0;
        ArrayList<Instance> D_ = new ArrayList<>();
        Instance inst;

        for (int j = i; j < D.numInstances(); j++) {
            inst = new DenseInstance(1, D.get(j).toDoubleArray());
            D_.add(inst);
        }
        return D_;
    }
    
    /**
     * 
     * @param train base de dados rotulada
     * @param classIndex qtde de classes do problema
     * @return retorna uma lista contendo as classes conhecidas (C_con) do problema.
     */
    public static Set<String> getClassesConhecidas(List<InstanceExample> train, int L){
        int[] dist = DataSetUtils.getLabelsDistribution(train, L);
        Set<String> classesConhecidas = new HashSet<>();
        for (int j = 0; j < train.size(); j++) {
            for (int k = 0; k < train.get(j).getData().numOutputAttributes(); k++) {
                if(train.get(j).getData().valueOutputAttribute(k) == 1 && dist[k] > 10){
                    classesConhecidas.add(String.valueOf(k));
                }
            }
        }
        return classesConhecidas;
    }
    
    public static ArrayList<double[]> getWindowsCardinalities(ArrayList<InstanceExample> dataSet, int windowsSize, int numWindows, int numLabels){
        ArrayList<double[]> cardinalities = new ArrayList<>() ;
        for (int i = 0; i < numWindows; i++) {
            try{
                if(i == numWindows-1)
                    cardinalities.add(getLabelFrequencies(dataSet.subList(i*windowsSize, dataSet.size())));
                else
                    cardinalities.add(getLabelFrequencies(dataSet.subList(i*windowsSize, (i*windowsSize)+windowsSize)));
            }catch(Exception e){
                e.printStackTrace();
                System.out.println("");
                System.exit(0);
            }
        }
        return cardinalities;
    }
    
    /**
     * returns label frequencies given a dataset
     * @param D - dataset
     * @return 
     */
    public static final double[] getLabelFrequencies(List<InstanceExample> D) {
		int L = D.get(0).getData().numOutputAttributes();
		double lc[] = new double[L];
		for(int j = 0; j < L; j++) {
			for(int i = 0; i < D.size(); i++) {
				lc[j] += D.get(i).getData().valueOutputAttribute(j);
			}
			lc[j] /= D.size();
		}
		return lc;
	}
    
    public static Set<String> getClassesConhecidas(ArrayList<weka.core.Instance> train, int L){
        Set<String> classesConhecidas = new HashSet<>();
        
        for (int j = 0; j < train.size(); j++) {
            for (int k = 0; k < L; k++) {
                if(train.get(j).value(k) == 1){
                    classesConhecidas.add(String.valueOf(k));
                }
            }
        }
        return classesConhecidas;
    }
    
    /**
     * Convert to the arrf file with classes first
     * @param dataSetName
     * @param dataSet
     * @throws IOException
     * @throws Exception 
     */
    public static void invertArff(String dataSetName, String dataSet) throws IOException, Exception {
        Instances D_ = DataSetUtils.dataFileToInstance(dataSet);
        int L = D_.classIndex();
        //output file
        FileWriter dataSetFile = new FileWriter(new File(dataSetName+"-V2.arff"), false);
        //write the file's header
        dataSetFile.write("@relation '"+dataSetName+": -C "+ L + "'\n");
        dataSetFile.write("\n");
        for (int i = 0; i < L; i++) {
            dataSetFile.write("@attribute class"+i+" {0, 1}\n");
        }
        for (int i = 0; i < D_.numAttributes()-L; i++) {
            dataSetFile.write("@attribute att"+i+" numeric\n");
        }
        dataSetFile.write("\n");
        dataSetFile.write("@data\n");
        for (int i = 0; i < D_.numInstances(); i++) {
            for (int j = 0; j < D_.numAttributes(); j++) {
                if(j < L){
                    dataSetFile.write(""+(int)D_.instance(i).value(j)+",");
                }else{
                    if(j ==  D_.numAttributes()-1){
                        dataSetFile.write(""+D_.instance(i).value(j));
                    }else{
                        dataSetFile.write(""+D_.instance(i).value(j)+",");
                    }
                }
            }
            dataSetFile.write("\n");
            System.out.println("\n");
        }
        dataSetFile.close();
    }
    
    
    /**
     * Lê um arquivo arff e transforma em um conjunto de exemplos
     * @param fileName data file path
     * @return conjunto de exemplos D
     */
    public static Instances dataFileToInstance(String fileName) throws Exception {
        ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource(fileName);
        Instances D = dataSource.getDataSet();
        MLUtils.prepareData(D);
        return D;
    }
    
   /**
    * Transforma uma linha de um arquivo em uma instância
    * @param line
    * @return 
    */ 
   public static Instance stringToInstance(String line){
       String[] values = line.split(",");
      
       ArrayList<Float> listaValuesDouble = new ArrayList<>();
       ArrayList<Integer> index = new ArrayList<>();
       for (int i = 0; i < values.length; i++) {
           if(i == 0){
               String[] aux = values[i].split(" ");
               index.add(Integer.parseInt(aux[0].substring(1)));
               listaValuesDouble.add(Float.parseFloat(aux[1]));
           }else if(i == values.length - 1){
               String[] aux = values[i].split(" ");
               String value = aux[1].substring(0, aux[1].length()-1);
               index.add(Integer.parseInt(aux[0]));
               listaValuesDouble.add(Float.parseFloat(value));
           }else{
               String[] aux = values[i].split(" ");
               index.add(Integer.parseInt(aux[0]));
               listaValuesDouble.add(Float.parseFloat(aux[1]));
           }
       }
       double[] valuesDouble = new double[index.get(index.size()-1) + 1];
       for (int i = 0; i < index.size(); i++) {
           valuesDouble[index.get(i)] = listaValuesDouble.get(i);
       }
       return new DenseInstance(1, valuesDouble);
   }
   
   
   /**
    * Verifica se há uma classe novidade entre as classes reais da nova instância
    * @param x
    * @param y
    * @return 
    */
   public static int hasNewClass(Instance x, Instance y){
       int numAttX = x.numAttributes();
       int numAttY = y.numAttributes();
       return numAttX - numAttY;
        
   }
   
       public static void ordenar(HashMap<String, ArrayList<Instance>> aux, FileWriter fileOff) throws IOException {
        String maior = null;
        int numMaior = 0;
        int auxSize = aux.size();
        for (int i = 0; i < auxSize; i++) {
            for (Map.Entry<String, ArrayList<Instance>> entry : aux.entrySet()) {
                String key = entry.getKey();
                ArrayList<Instance> value = entry.getValue();
                if (value.size() > numMaior) {
                    numMaior = value.size();
                    maior = key;
                }
            }
            fileOff.write("Combinação: " + maior + " #SUP: " + numMaior + "\n");
            System.out.println("Combinação: " + maior + "Qtde Exemplos: " + numMaior + "\n");
            aux.remove(maior);
            numMaior = 0;
            maior = null;
        }
        fileOff.close();
    }
    
     /**
     * Split the dataset into train and test
     * @param train
     * @param test
     * @param file
     * @param numInstances 
     */
    public static void slipTrainTest(ArrayList<Instance> train, ArrayList<Instance> test, MultiTargetArffFileStream file, int numInstances, double percent){
        for (int i = 0; i < numInstances; i++) {
//            Instance instance = file.nextInstance().getData();
            if(i < numInstances*percent)
                train.add(file.nextInstance().getData());
            else
                test.add(file.nextInstance().getData());
        }
    }
     /**
     * Split the dataset into train and test
     * @param train
     * @param test
     * @param file
     * @param numInstances 
     */
    public static void slipTrainTestStratification(ArrayList<Instance> train, ArrayList<Instance> test, MultiTargetArffFileStream file, int numInstances, double percent, int L){
        HashMap<String, Set<Integer>> stratifiedList = new HashMap<String, Set<Integer>>();
        for (int i = 0; i < L; i++) {
            Set<Integer> set = new HashSet<Integer>();
            stratifiedList.put(""+i, set);
        }
        
        for (int i = 0; i < numInstances; i++) {
            if(i < numInstances*percent)
                train.add(file.nextInstance().getData());
            else
                test.add(file.nextInstance().getData());
        }
    }
     /**
     * Split the dataset into train and test
     * @param train
     * @param test
     * @param file
     * @param numInstances 
     */
    public static void slipTrainTestValidation(ArrayList<Instance> train, ArrayList<Instance> test, MultiTargetArffFileStream file, int numInstances){
        for (int i = 0; i < numInstances*0.2; i++) {
            if(i < numInstances*0.1)
                train.add(file.nextInstance().getData());
            else
                test.add(file.nextInstance().getData());
        }
    }
    
    
    public static HashMap<String, Float> getDependencesDegree(ArrayList<Instance> dataSet){
        HashMap<String, int[][]> contingencyTableList = new HashMap<String, int[][]>();
        int L = dataSet.get(0).numOutputAttributes();
        for (Instance instance : dataSet) {
            int[] Y = getBipartition(instance, L);
            for (int i = 0; i < L-1; i++) {
                for (int j = i+1; j < L; j++) {
                    String y = i + "," + j;
                    int[][] mtx = contingencyTableList.get(y);
                    if(contingencyTableList.get(y) == null){
                        mtx = new int[2][2];
                    }
                    if(Y[i] == 1 && Y[j] == 1)
                        mtx[0][0] += 1;
                    else if(Y[i] == 0 && Y[j] == 0)
                        mtx[1][1] += 1;
                    else if(Y[i] == 1 && Y[j] == 0)
                        mtx[1][0] += 1;
                    else if(Y[i] == 0 && Y[j] == 1)
                        mtx[0][1] += 1;
                    else
                        System.out.println("Deu ruim");
                    contingencyTableList.put(y, mtx);
//                    y = j + "," + i;
//                    mtx = contingencyTableList.get(y);
//                    if(contingencyTableList.get(y) == null){
//                        mtx = new int[2][2];
//                    }
//                    if(Y[j] == 1 && Y[i] == 1)
//                        mtx[0][0] += 1;
//                    else if(Y[j] == 0 && Y[i] == 0)
//                        mtx[1][1] += 1;
//                    else if(Y[j] == 1 && Y[i] == 0)
//                        mtx[1][0] += 1;
//                    else if(Y[j] == 0 && Y[i] == 1)
//                        mtx[0][1] += 1;
//                    else
//                        System.out.println("Deu ruim");
//                    contingencyTableList.put(y, mtx);
                }
            }
        }
//        int[][] dependenceScoreList = getChiSquareScores(contingencyTableList, L);
//        float[][] dependenceScoreList = new float[L][L];
//        HashMap<String, Float> pairwiseResult = getScores(contingencyTableList, L);
//        for (int i = 0; i < L; i++) {
//            for (int j = 0; j < L; j++) {
//                if()
//                pairwiseResult.put(i+","+j, dependenceScoreList[i][j]);
//            }
//        }
//        for (int i = 0; i < dependenceScoreList.length; i++) {
//            for (int j = 0; j < dependenceScoreList.length; j++) {
//                System.out.print(dependenceScoreList[i][j]+ " ");
//            }
//            System.out.print("\n");
//        }
//        System.out.println("");
        return  getScores(contingencyTableList, L);
    }
    
    
    public static HashMap<int[], Double> calculateUncondionalDependence(ArrayList<Instance> dataSet){
        HashMap<String, int[][]> contingencyTableList = new HashMap<String, int[][]>();
        int L = dataSet.get(0).numOutputAttributes();
        for (Instance instance : dataSet) {
            int[] Y = getBipartition(instance, L);
            for (int i = 0; i < L-1; i++) {
                for (int j = i+1; j < L; j++) {
                    String y = i + "," + j;
                    int[][] mtx = contingencyTableList.get(y);
                    if(contingencyTableList.get(y) == null){
                        mtx = new int[2][2];
                    }
                    if(Y[i] == 1 && Y[j] == 1)
                        mtx[0][0] += 1;
                    else if(Y[i] == 0 && Y[j] == 0)
                        mtx[1][1] += 1;
                    else if(Y[i] == 1 && Y[j] == 0)
                        mtx[1][0] += 1;
                    else
                        mtx[0][1] += 1;
                    contingencyTableList.put(y, mtx);
                }
            }
        }
        int[][] dependenceScoreList = getChiSquareScores(contingencyTableList, L);
        for (int i = 0; i < dependenceScoreList.length; i++) {
            for (int j = 0; j < dependenceScoreList.length; j++) {
                System.out.print(dependenceScoreList[i][j]+ " ");
            }
            System.out.print("\n");
        }
        System.out.println("");
        return null;
        
    }
    
    public static void sumContigencyTable(int[][] mtx){
        mtx[2][0] = mtx[0][0] + mtx[1][0]; //a + C
        mtx[2][1] = mtx[0][1] + mtx[1][1]; //b + d
        mtx[0][2] = mtx[0][0] + mtx[0][1]; //a+b
        mtx[1][2] = mtx[1][0] + mtx[1][1]; //c+d
        mtx[2][2] = mtx[0][0] + mtx[0][1] + mtx[1][0] + mtx[1][1]; //a+b+c+d
    }
    
    private static int[][] getChiSquareScores(HashMap<String, int[][]>  dependenceScoreList, int L){
        int[][] scoresMtx = new int[L][L];
        for (Map.Entry<String, int[][]> entry : dependenceScoreList.entrySet()) {
            String key = entry.getKey();
            int[][] value = entry.getValue();
            for (int i = 0; i < value.length; i++) {
                for (int j = 0; j < value.length; j++) {
                    System.out.print(value[i][j] + " ");
                }
                System.out.print("\n");
            }
            int a = value[0][0];
            int b = value[0][1];
            int c = value[1][0];
            int d = value[1][1];
            double N = a+b+c+d;
            double expected = Math.pow((a*d) - (b*c), 2) * N;
            double score =   expected / (a+b)*(c+d)*(b+d)*(a+c);
            String[] aux = key.split(",");
            scoresMtx[Integer.parseInt(aux[0])][Integer.parseInt(aux[1])] = a;
            scoresMtx[Integer.parseInt(aux[1])][Integer.parseInt(aux[0])] = a;
        }
        return scoresMtx;
    }
    
    private static HashMap<String, Float> getScores(HashMap<String, int[][]>  dependenceScoreList, int L){
        HashMap<String, Float> scores = new HashMap<>();
        for (Map.Entry<String, int[][]> entry : dependenceScoreList.entrySet()) {
            String key = entry.getKey();
            int[][] value = entry.getValue();
            int a = value[0][0];
            int b = value[0][1];
            int c = value[1][0];
            int d = value[1][1];
            float score = testJ(a, b, c, d);
//            float score = testPr(a, b);
//            float score = testF1(a, b, c, d);
            String[] aux = key.split(",");
            scores.put(key, score);
//            scoresMtx[Integer.parseInt(aux[1])][Integer.parseInt(aux[0])] = score;
        }
        return scores;
    }
    
    /**
     * Calculates, based on a contingency table, the F1 value between two classes
     * @param a
     * @param b
     * @param c
     * @param d
     * @return 
     */
    private static float testF1(int a, int b, int c, int d) {
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
    private static float testJ(int a, int b, int c, int d) {
        float j = 0;
        if(a > 0 && (b > 0 || c > 0)){
            j = (float)a / (float)(a + b + c);
        }
        return j;
    }
    
    private static float testPr(int a, int b) {
        float j = ((float)a / (float)(a+b));
        return j;
    }
    
    /**
     * 
     * @param D data
     * @param novelties novelty classes index
     * @param maxSupportTrain maximal number of examples of a specific class
     * @param trainSizePercent 
     * @throws Exception 
     */
    public static void noveltyDetectionStratificationNusWide(List<Instance> D, List<Instance> train, List<Instance> test, Set<String> novelties, int maxSupportTrain, double trainSizePercent) throws Exception{
        int[] control = new int[D.get(0).numOutputAttributes()];
        int contTrain = 0;
        int trainSize = (int) (D.size()*trainSizePercent);
        HashMap<String, ArrayList<Instance>> noveltyInstances = new HashMap<>();
        ArrayList<Instance> normalInstances = new ArrayList<>();
        int index = 0;
        for (int i = 0; i < D.size(); i++) {
            Instance inst = D.get(i);
            if(contTrain <= trainSize){
                Set<String> Y = DataSetUtils.getLabelSet(inst);
                Set<String> Y_aux = new HashSet<>();
                Y_aux.addAll(Y);
                Y_aux.retainAll(novelties);
                boolean flag = true;
                if(!Y_aux.isEmpty()){
                    ArrayList<Instance> arrayList = noveltyInstances.get(Y_aux.toString());
                    if(arrayList == null){
                       arrayList = new ArrayList<>();
                    }
                    arrayList.add(inst);
                    noveltyInstances.put(Y_aux.toString(), arrayList);
                }else{
                    for (String y : Y) {
                        if(control[Integer.parseInt(y)] >= maxSupportTrain){
                            normalInstances.add(inst);
                            flag = false;
                            break;
                        }
                    }
                    if(flag){
                        for (String y : Y) {
                            control[Integer.parseInt(y)] += 1;
                        }
                        train.add(inst);
                        contTrain++;
                    }
                }
            }else{
                index = i;
                break;
            }
        }
        if(index > 0){
           for (int i = index; i < D.size(); i++) {
                normalInstances.add(D.get(i));
            } 
        }
        
        int cont2 = normalInstances.size()/2;
        for (int i = 0; i < normalInstances.size()/2; i++) {
            test.add(normalInstances.get(i));
        }
        
        for (Map.Entry<String, ArrayList<Instance>> entry : noveltyInstances.entrySet()) {
            ArrayList<Instance> value = entry.getValue();
            for (Instance instance : value) {
//                if(!test.contains(instance)){
                    test.add(instance);
//                }
            }
        }
        for (int i = cont2; i < normalInstances.size(); i++) {
            test.add(normalInstances.get(i));
        }
    }
    
    public static void noveltyDetectionStratification(List<Instance> D, List<Instance> train, List<Instance> test, Set<String> novelties, int maxSupportTrain, double trainSizePercent) throws Exception{
        int[] control = new int[D.get(0).numOutputAttributes()];
        int contTrain = 0;
        int trainSize = (int) (D.size()*trainSizePercent);
        HashMap<String, ArrayList<Instance>> noveltyInstances = new HashMap<>();
        ArrayList<Instance> normalInstances = new ArrayList<>();
        int index = 0;
        for (int i = 0; i < D.size(); i++) {
            Instance inst = D.get(i);
            if(contTrain <= trainSize){
                Set<String> Y = DataSetUtils.getLabelSet(inst);
                Set<String> Y_aux = new HashSet<>();
                Y_aux.addAll(Y);
                Y_aux.retainAll(novelties);
                boolean flag = true;
                if(!Y_aux.isEmpty()){
                    ArrayList<Instance> arrayList = noveltyInstances.get(Y_aux.toString());
                    if(arrayList == null){
                       arrayList = new ArrayList<>();
                    }
                    arrayList.add(inst);
                    noveltyInstances.put(Y_aux.toString(), arrayList);
                }else{
                    for (String y : Y) {
                        if(control[Integer.parseInt(y)] >= maxSupportTrain){
                            normalInstances.add(inst);
                            flag = false;
                            break;
                        }
                    }
                    if(flag){
                        for (String y : Y) {
                            control[Integer.parseInt(y)] += 1;
                        }
                        train.add(inst);
                        contTrain++;
                    }
                }
            }else{
                index = i;
                break;
            }
        }
        if(index > 0){
           for (int i = index; i < D.size(); i++) {
                normalInstances.add(D.get(i));
            } 
        }
        int cont2 = normalInstances.size()/noveltyInstances.size();
        index = 0;
//        ArrayList<Instance> auxTest = new ArrayList<>();
        for (Map.Entry<String, ArrayList<Instance>> entry : noveltyInstances.entrySet()) {
            ArrayList<Instance> value = entry.getValue();
            for (Instance instance : value) {
                if(!test.contains(instance)){
                    test.add(instance);
                }
            }
//            if(cont2 < normalInstances.size()){
                for (int i = index; i < cont2; i++) {
//                    try{
                        test.add(normalInstances.get(i));
//                    }catch(Exception e){
//                        System.out.println("");
//                    }
                }
                index = cont2;
                cont2 += normalInstances.size()/noveltyInstances.size();
//            }else{
//                System.out.println("");
//            }
        }
        for (int i = index; i < normalInstances.size(); i++) {
            test.add(normalInstances.get(i));
        }
    }

    public static void createFileBasedOnList(List<Instance> list, String dataSetPath, String name, String dataSetName) throws IOException {
        String fileName = dataSetPath.replace(".arff", "_"+name+".arff");
        int m = list.get(0).numOutputAttributes();
        FileWriter file = new FileWriter(new File(fileName), false);
        file.write("@relation '" + dataSetName+"_"+name + ": -C "+ m + "'\n");
        file.write("\n");
        
        for (int i = 0; i < m; i++) {
            file.write("@attribute "+list.get(0).outputAttribute(i).name()+" {0, 1}\n");
        }

        int numAtt = list.get(0).numInputAttributes();
        
        for (int i = 0; i < numAtt; i++) {
            file.write("@attribute " + list.get(0).inputAttribute(i).name() + " numeric\n");
        }
        file.write("\n");
        file.write("@data\n");
        
        for (int i = 0; i < list.size(); i++) {
            for (int j = 0; j < m; j++) {
                file.write(((int)list.get(i).value(j))+",");
            }
            for (int j = 0; j < numAtt; j++) {
                if(j < numAtt-1)
                    file.write(list.get(i).valueInputAttribute(j)+",");
                else
                    file.write(list.get(i).valueInputAttribute(j)+"\n");
            }
        }
        file.close();
    }
    
}
