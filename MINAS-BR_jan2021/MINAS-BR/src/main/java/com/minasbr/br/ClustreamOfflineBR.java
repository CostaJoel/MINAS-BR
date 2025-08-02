package com.minasbr.br;

import NoveltyDetection.ClustreamKernelMOAModified;
import NoveltyDetection.ClustreamMOAModified;
import NoveltyDetection.KMeansMOAModified;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

import moa.cluster.Clustering;

public class ClustreamOfflineBR {

    private ArrayList<Integer> clusteringResul[];
    private int clusterExamples[];

    /**
     * 
     * @param examples
     * @param qtdeTotalClasses
     * @param numMicro
     * @param flagMicro
     * @param flagKMeans
     * @return
     * @throws IOException 
     */   
    public Clustering CluStream(ArrayList<Instance> examples, int qtdeTotalClasses, float numMicro, boolean flagMicro, boolean flagKMeans) throws IOException {
        Clustering microClusters = null;

        ClustreamMOAModified algClustream = new ClustreamMOAModified();
        algClustream.prepareForUse();

        if (flagKMeans == true) {
            algClustream.initializeBufferSize(2 * (int) numMicro, (int) numMicro);
        } else {
            algClustream.initializeBufferSize(1 * (int) numMicro, (int) numMicro);
        }

        //read the data set and executing clustream algorithm
        for (int i = 0; i < examples.size(); i++) {
            //pega os atributos sem as classes
            double[] data = Arrays.copyOfRange(examples.get(i).toDoubleArray(), qtdeTotalClasses, examples.get(i).numAttributes());
            Instance inst = new DenseInstance(1, data);
            algClustream.trainOnInstanceImpl(inst);
        }
       
        //obtain the micro-clusters  
        microClusters = algClustream.getMicroClusteringResult();
        ArrayList<ArrayList<Integer>> grupos = new ArrayList<ArrayList<Integer>>();
        ArrayList<double[]> centers = new ArrayList<double[]>();

        if (flagMicro == false) { //using clustream + KMeans       	
            //choose the best value for K and executing KMeans
            Clustering macroClustering = null;
            macroClustering = KMeansMOAModified.OMRk(microClusters, 0, grupos);
            int numClusters = macroClustering.size();

            ArrayList<ArrayList<Integer>> grupoMicro = new ArrayList<ArrayList<Integer>>();
            grupoMicro = ((ClustreamMOAModified) algClustream).getClusterExamples();
            clusteringResul = (ArrayList<Integer>[]) new ArrayList[numClusters];

            for (int i = 0; i < clusteringResul.length; i++) {
                clusteringResul[i] = new ArrayList<Integer>();
            }

            for (int i = 0; i < grupos.size(); i++) {
                for (int j = 0; j < grupos.get(i).size(); j++) {
                    int pos = grupos.get(i).get(j);
                    for (int k = 0; k < grupoMicro.get(pos).size(); k++) {
                        clusteringResul[i].add((int) grupoMicro.get(pos).get(k));
                    }
                }
            }
            for (int k = 0; k < numClusters; k++) {
                centers.add(macroClustering.getClustering().get(k).getCenter());
            }

        } else { // using only clustream
            if (microClusters.size() >= 1) {
                grupos = ((ClustreamMOAModified) algClustream).getClusterExamples();

                clusterExamples = new int[examples.size()];
                int value;
                for (int i = 0; i < grupos.size(); i++) {
                    //remove micro-cluster with less than 3 examples
                    if (((ClustreamKernelMOAModified) microClusters.get(i)).getWeight() < 3) {
                        value = -1;
                    } else {
                        value = i;
                    }

                    for (int j = 0; j < grupos.get(i).size(); j++) {
                        clusterExamples[grupos.get(i).get(j)] = value; //i+1;
                    }
                    if (((ClustreamKernelMOAModified) microClusters.get(i)).getWeight() < 3) {
                        microClusters.remove(i);
                        grupos.remove(i);
                        i--;
                    }
                }
            }
        }
        return microClusters;
    }

//*******************************************************************************   
//******************** getClusterExamples ***************************************
//******************************************************************************* 	
    public int[] getClusterExamples() {
        return clusterExamples;
    }

//*******************************************************************************   
//******************** Clustering ***********************************************
//******************************************************************************* 	
    public int[] getClusteringResults() {
        int resultado[];
        int nroelem = 0;
        for (int i = 0; i < clusteringResul.length; i++) {
            nroelem += clusteringResul[i].size();
        }
        resultado = new int[nroelem];

        for (int i = 0; i < clusteringResul.length; i++) {
            for (int j = 0; j < clusteringResul[i].size(); j++) {
                resultado[clusteringResul[i].get(j)] = i + 1;
            }
        }
        return resultado;
    }
}
