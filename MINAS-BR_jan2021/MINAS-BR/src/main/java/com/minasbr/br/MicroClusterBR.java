/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.minasbr.br;

import NoveltyDetection.KMeansMOAModified;
import NoveltyDetection.MicroCluster;
import com.yahoo.labs.samoa.instances.Instance;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;

/**
 *
 * @author joel
 */
public class MicroClusterBR {
    private double threshold;
    private double averOut;
    private MicroCluster microCluster;
    private double silhouette;

    public double getSilhouette() {
        return silhouette;
    }

    public MicroClusterBR(MicroCluster microCluster) {
        this.microCluster = microCluster;
    }

    /**
     * @return the threshold
     */
    public double getThreshold() {
        return threshold;
    }

    /**
     * @return the averOut
     */
    public double getAverOut() {
        
        return averOut;
    }

    /**
     * @return the microCluster
     */
    public MicroCluster getMicroCluster() {
        return microCluster;
    }
    
    /**
     * Silhouette validation
     *
     * @param modeloValidar model
     * @return
     */
    public double clusterSilhouette(ArrayList<MicroClusterBR> modeloValidar) {
        double minDistance = Double.MAX_VALUE;
        // calculate the distance between the center of the new cluster to the existing clusters 
        for (int i = 1; i < modeloValidar.size(); i++) {
            double distance = KMeansMOAModified.distance(modeloValidar.get(i).getMicroCluster().getCenter(), this.microCluster.getCenter());
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
        this.silhouette = (minDistance - this.microCluster.getRadius() / 2) / Math.max(minDistance, this.microCluster.getRadius() / 2);
        
        return this.silhouette;
    }

    public void calculateInitialAverOutput(ArrayList<double[]> X_b) {
        double sum = 0;
        sum = X_b.stream()
                .map(x_i -> Math.exp(-KMeansMOAModified.distance(this.getMicroCluster().getCenter(), x_i)))
                .reduce(sum, (accumulator, _item) -> accumulator + _item);
        this.setAverOut(sum / X_b.size());
    }
    
    public void calculateAverOutputNovelty(List<Entry<Integer, Instance>> toClassify){
        ArrayList<double[]> toCalculateAvgOut = new ArrayList<>();

        toClassify.forEach(inst -> {
            toCalculateAvgOut.add(Arrays.copyOfRange(inst.getValue().toDoubleArray(),
                    inst.getValue().numOutputAttributes(),
                    inst.getValue().numAttributes()));
        });
        this.calculateInitialAverOutput(toCalculateAvgOut);
    }

    public void calculateThreshold(HashMap<String, Integer> mtxLabelsFrequencies, double observedExamples) {
        String j = this.getMicroCluster().getLabelClass();
        int yj = 0;
        try{
            yj = mtxLabelsFrequencies.get(j + "," + j);
        }catch(NullPointerException e){
            System.err.println("[Error] calculateThreshold - mtxLabelsFrequencies doesn't has cordinate keys");
            e.printStackTrace();
            System.exit(0);
        }
        double p_yj =  yj / observedExamples;
        double prod = 1;
        double prod1 = 1;
        for (Map.Entry<String, Integer> entry : mtxLabelsFrequencies.entrySet()) {
            String key[] = entry.getKey().split(",");
            if(!key[0].equals(j) && key[1].equals(j)){
                double p_yk_yj = (double) entry.getValue() / (double) yj;
                prod *= p_yk_yj;
//                prod1 *= p_yk_yj * this.averOut;
            }
        }
        this.setThreshold(p_yj * prod * this.averOut);
        if (this.threshold > 1){
            System.err.println("Threshold > 1");
            System.exit(0);
        }
//        this.threshold = p_yj * prod1;
    }

    public void updateAverOut(double dist) {
        this.setAverOut(((double) this.getMicroCluster().getN() * this.averOut + Math.exp(-dist)) / (double) (this.getMicroCluster().getN() + 1));
        if(this.averOut > 1){
            System.err.println("AverOut > 1");
            System.exit(0);
        }
//        this.averOut += exp_dist;
    }

    /**
     * @param threshold the threshold to set
     */
    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    /**
     * @param averOut the averOut to set
     */
    public void setAverOut(double averOut) {
        this.averOut = averOut;
    }

}
