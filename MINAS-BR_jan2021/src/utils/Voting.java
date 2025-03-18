/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utils;

import br.Model;
import java.util.ArrayList;

/**
 * Classe usada para receber a votação dos classisificadores
 * @author joel
 */
public class Voting implements Comparable<Voting>{
    private String category;
    private String label;
    private double distance;
    private double threashold;
    private String tipoNov;
    private int posMC;

    public void setPosMC(int posMC) {
        this.posMC = posMC;
    }

    public Voting(String category, String label, double distance, int posMC, double threashold) {
        this.category = category;
        this.label = label;
        this.distance = distance;
        this.threashold = threashold;
        this.posMC = posMC;
    }
    
    public Voting() {
    }
    
    /**
     * Separa os modelos entre quem considera o micro-grupo como extensão e quem considera pn
     * @param votacao lista com os resultados
     * @return lista 1 - modelos extensão; lista 2 - modelos pn
     */
    public static ArrayList<ArrayList<Voting>> getSeparateModels(ArrayList<Voting> votacao){
        ArrayList<Voting> extModels = new ArrayList<Voting>();
        ArrayList<Voting> pnModels = new ArrayList<Voting>();
        ArrayList<ArrayList<Voting>> separateModels = new ArrayList<ArrayList<Voting>>();
        for (Voting voting : votacao) {
            if(voting.getDistance() < voting.getThreashold()){
                extModels.add(voting);
            }else{
                pnModels.add(voting);
            }
        }
        separateModels.add(extModels);
        separateModels.add(pnModels);
        return separateModels;
    }
    

    /**
     * @return the category
     */
    public String getCategory() {
        return category;
    }

    /**
     * @param category the category to set
     */
    public void setCategory(String category) {
        this.category = category;
    }

    /**
     * @return the key
     */
    public String getlabel() {
        return label;
    }

    /**
     * @param key the key to set
     */
    public void setlabel(String label) {
        this.label = label;
    }

    /**
     * @return the distance
     */
    public double getDistance() {
        return distance;
    }

    /**
     * @param distance the distance to set
     */
    public void setDistance(double distance) {
        this.distance = distance;
    }

    @Override
    public int compareTo(Voting o) {
        if (this.distance < o.getDistance()) {
          return -1;
        }
        if (this.distance > o.getDistance()) {
             return 1;
        }
        return 0;
    }

    /**
     * @return the threashold
     */
    public double getThreashold() {
        return threashold;
    }

    /**
     * @param threashold the threashold to set
     */
    public void setThreashold(double threashold) {
        this.threashold = threashold;
    }

    /**
     * @return the tipoNov
     */
    public String getTipoNov() {
        return tipoNov;
    }

    /**
     * @param tipoNov the tipoNov to set
     */
    public void setTipoNov(String tipoNov) {
        this.tipoNov = tipoNov;
    }

    /**
     * @return the posMC
     */
    public int getPosMC() {
        return posMC;
    }

}
