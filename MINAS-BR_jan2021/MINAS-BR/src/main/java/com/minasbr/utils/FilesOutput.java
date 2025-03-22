/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.minasbr.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 *
 * @author joel
 */
public class FilesOutput {
    
    /**
     * Cria um diretório no caminho indicado
     * @param outputDirectory 
     */
    public static void createDirectory(String outputDirectory) {
        File dir = new File(outputDirectory);
        if (dir.exists()) {
            dir.delete();
        } else if (dir.mkdirs()) {
            System.out.println("Directory created successfully");
        }
    }
    
    /**
     * Cria arquivos de saída dentro da devida pasta
     * @param outputDirectory  caminho da pasta que deve ser armazenado os resultados
     * @param fileOut  armazena os resultados
     * @param fileOutClasses armazena as classes do problema
     * @throws IOException 
     */
    public static FileWriter createFileOut(String outputDirectory) throws IOException {
        //String fileNameOut = outputDirectory + "\\results";
        String fileNameOut = outputDirectory + "/results";
        //String fileNameOutClasses = outputDirectory + "\\results" + "Classes";
//        String fileNameOutClasses = outputDirectory + "/results" + "Classes";
        FileWriter fileOut = new FileWriter(new File(fileNameOut), false);
//        fileOutClasses = new FileWriter(new File(fileNameOutClasses), false);

        fileOut.write("Results");
        fileOut.write("\n\n \n\n");
        
        return fileOut;
    }
}
