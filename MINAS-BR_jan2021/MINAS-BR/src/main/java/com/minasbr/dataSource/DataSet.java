/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.minasbr.dataSource;

import java.util.ArrayList;
import java.util.HashMap;

/**
 *
 * @author joel
 */
public class DataSet {
    private final String name;
    private final String path;
    private final String outputDir;
    private HashMap<Integer, String> listaMarcacoes;

    public DataSet(String name, String path, String outputDir, HashMap<Integer, String> listaMarcacoes) {
        this.name = name;
        this.path = path;
        this.outputDir = outputDir;
        this.listaMarcacoes = listaMarcacoes;
    }

    /**
     * @return the name
     */
    public String getName() {
        return name;
    }

    /**
     * @return the path
     */
    public String getPath() {
        return path;
    }

    /**
     * @return the listaMarcacoes
     */
    public HashMap<Integer, String> getListaMarcacoes() {
        return listaMarcacoes;
    }

    /**
     * @return the outputDir
     */
    public String getOutputDir() {
        return outputDir;
    }

    
}
