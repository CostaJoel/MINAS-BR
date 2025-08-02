/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.minasbr.utils;

import com.yahoo.labs.samoa.instances.Instance;
import java.util.ArrayList;

/**
 *
 * @author bioinfo02
 */
public class ShortTimeMemory {
    private ArrayList<Instance> data;
    private ArrayList<Integer> timestamp;
    private int qtdeExDeleted;
    
    /**
     * Add a element in the short time memory
     * @param x instance to add
     * @param timestamp 
     */
    public void add(Instance x, int timestamp){
        this.getData().add(x);
        this.getTimestamp().add(timestamp);
    }

    public ShortTimeMemory(ArrayList<Instance> data, ArrayList<Integer> timestamp) {
        this.data = data;
        this.timestamp = timestamp;
    }
    
    /**
     * Get the number of elements in the short time memory
     * @return the size
     */
    public int size(){
        assert this.getData().size() == this.getTimestamp().size();
        return this.getData().size();
    }

    /**
     * @return the data
     */
    public ArrayList<Instance> getData() {
        return data;
    }

    /**
     * @return the timestamp
     */
    public ArrayList<Integer> getTimestamp() {
        return timestamp;
    }
    
    /**
     * Remove exemplo da memória temporária
     * @param index 
     */
    public void remove(int index){
        this.data.remove(index);
        this.timestamp.remove(index);
    }

    /**
     * @return the qtdeExDeleted
     */
    public int getQtdeExDeleted() {
        return qtdeExDeleted;
    }

    /**
     * @param qtdeExDeleted the qtdeExDeleted to set
     */
    public void setQtdeExDeleted(int qtdeExDeleted) {
        this.qtdeExDeleted = qtdeExDeleted;
    }
}
