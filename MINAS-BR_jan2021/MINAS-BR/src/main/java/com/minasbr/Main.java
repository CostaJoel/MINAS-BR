package com.minasbr;

import com.minasbr.baselines.batch.ExperimentBatch;
import com.minasbr.baselines.upperbound.ExperimentUpperBound;
import com.minasbr.br.Model;
import com.minasbr.br.OfflinePhase;
import com.minasbr.br.OnlinePhase;
import com.yahoo.labs.samoa.instances.Instance;
import com.minasbr.dataSource.DataSetUtils;
import com.minasbr.evaluate.Evaluator;
import com.minasbr.evaluate.EvaluatorBR;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import moa.streams.MultiTargetArffFileStream;
import com.minasbr.utils.FilesOutput;
import weka.core.Instances;

public class Main {

    public static void main(String[] args) throws Exception {

        if (args.length == 0) {
            System.out.println("Nenhum argumento foi passado!");
            System.out.println("Lista de argumentos:");
            System.out.println("0: Tipo de algoritmo que será executando (MINAS-BR; upperbounds; lowerbounds)");
            System.out.println("1: Dataset name (e.g. 4CRE-V2, MOA1, etc.)");
            System.out.println("2: Caminho dataset de treino (e.g. src/main/resources/datasets/4CRE-V2-train.arff)");
            System.out.println("3: Caminho dataset de teste (e.g. src/main/resources/datasets/4CRE-V2-test.arff)");
            System.out.println("4: Quando args[0] = 'MINAS-BR', quantidade de classes no dataset (e.g. 4, 5, etc.)");
            System.out.println("5: Quando args[0] = 'MINAS-BR', parametro k_ini (N K inicial fase de treino)");
            System.out.println("6: Quando args[0] = 'MINAS-BR', parametro theta. Limite da memória temporária. O valor passado deve ser uma porcentagem, que representa uma porcentagem do tamanho janela de esquecimento de exemplos. (E.g. omega = 2000, theta = 0.5, então limiteMemoriaTemporaria = 2000*0.5, ou seja limiteMemoriaTemporaria = 1000 )");
            System.out.println("7: Quando args[0] = 'MINAS-BR', parametro omega (tamanho da janela de esquecimento de exemplos e de micro-clusters)");
            return;
        }

        // String tipoAlgoritmo = "upperbounds";
        // String tipoAlgoritmo = "lowerbounds";
        // String tipoAlgoritmo = "minas-br";
        // String dataSetName = "4CRE-V2";
        // int L = 4;
        // String trainPath = "resources/datasets/"+dataSetName+"-train.arff";
        // String testPath = "resources/datasets/"+dataSetName+"-train.arff";
        String tipoAlgoritmo = args[0].toLowerCase();
        String dataSetName = args[1];
        String trainPath = args[2];
        String testPath = args[3];
        String outputDirectory = "resources/results_output/" + dataSetName + "/" + tipoAlgoritmo;
        switch (tipoAlgoritmo) {
            case "minas-br":
                System.out.println("Você escolheu executar o MINAS-BR.");
                // double k_ini = 0.05;
                // String omega = "2000";
                // String theta = "" + (int) Math.ceil(Double.parseDouble("0.75") *
                // Double.parseDouble(omega));
                int L = Integer.parseInt(args[4]);
                double k_ini = Double.parseDouble(args[5]);
                String theta = "" + (int) Math.ceil(Double.parseDouble(args[6]) * Double.parseDouble(args[7]));
                String omega = args[7];
                System.out.println("Configuração de parametros:");
                System.out.println("Classes (L): " + L);
                System.out.println("Qtde de micro-clusters iniciais (k_ini): " + k_ini + "%");
                System.out.println("Limite da memória temporária (theta): " + theta);
                System.out.println("Tamanho da janela (omega): " + theta);
                experimentsMethods(trainPath,
                        testPath,
                        outputDirectory,
                        L,
                        k_ini,
                        theta,
                        omega,
                        "1.1",
                        "kmeans+leader",
                        "JI");
                // experimentsParameters(dataSetName, trainPath,
                // testPath,
                // L,
                // outputDirectory);
                break;
            case "upperbounds":
                System.out.println("Você escolheu executar os métodos Upper Bounds.");
                ExperimentUpperBound.execute(trainPath, testPath, outputDirectory, 50);
                break;
            case "lowerbounds":
                System.out.println("Você escolheu executar os métodos Lower Bounds.");
                ExperimentBatch.execute(trainPath, testPath, outputDirectory, "50");
                break;
            default:
                throw new Exception(
                        "Nome tipo algoritmo invalido! Selecionar entre as 3 opções: (MINAS-BR; upperbounds; lowerbounds)");
        }
        System.out.println("Execução do " + tipoAlgoritmo + " finalizada com sucesso!");
    }

    private static void experimentsParameters(String dataSetName,
            String trainPath,
            String testPath,
            int L,
            String outputDirectory) throws IOException, Exception {

        ArrayList<Instance> train = new ArrayList<Instance>();
        ArrayList<Instance> test = new ArrayList<Instance>();

        MultiTargetArffFileStream file = new MultiTargetArffFileStream(trainPath, String.valueOf(L));
        file.prepareForUse();
        while (file.hasMoreInstances()) {
            train.add(file.nextInstance().getData());
        }
        file.restart();

        file = new MultiTargetArffFileStream(testPath, String.valueOf(L));
        file.prepareForUse();

        while (file.hasMoreInstances()) {
            test.add(file.nextInstance().getData());
        }
        file.restart();
        // double[] theta = {1};
        // int[] omega = {2000};
        // double[] f = {1.1};
        // double[] k_ini = {0.01,0.001};
        double[] theta = { 0.1, 0.25, 0.5, 0.75, 1 };
        int[] omega = { 200, 500, 1000, 2000 };
        double[] f = { 1.1 };
        double[] k_ini = { 0.01, 0.05, 0.1, 0.25 };

        outputDirectory = outputDirectory + "/parameters_sensitivity/" + dataSetName + "/";
        FilesOutput.createDirectory(outputDirectory);

        FileWriter fileResults = new FileWriter(new File(outputDirectory + "/fullResults.csv"), false);
        fileResults.write(
                "shortTermMemoryLimit,windowsSize,threshold,k_ini,F1M,precision,recall,subsetAccuracy,NP,unk,unkRemoved\n");
        fileResults.close();

        for (int i = 0; i < theta.length; i++) {
            for (int j = 0; j < omega.length; j++) {
                if (theta[i] <= omega[j]) {
                    for (int k = 0; k < f.length; k++) {
                        for (int l = 0; l < k_ini.length; l++) {
                            String dir = outputDirectory + theta[i] + "_" + omega[j] + "_" + f[k] + "_" + k_ini[l]
                                    + "/";
                            System.out.println("***********" + theta[i] + "_" + omega[j] + "_" + f[k] + "_" + k_ini[l]
                                    + "******************");
                            FilesOutput.createDirectory(dir);
                            EvaluatorBR avMINAS = MINAS_BR(train,
                                    test,
                                    L,
                                    k_ini[l],
                                    (int) theta[i] * omega[j],
                                    omega[j],
                                    f[k],
                                    dir);

                            fileResults = new FileWriter(new File(outputDirectory + "/fullResults.csv"), true);

                            fileResults.write(theta[i] + "," +
                                    omega[j] + "," +
                                    f[k] + "," +
                                    k_ini[l] + "," +
                                    avMINAS.getAvgF1M() + "," +
                                    avMINAS.getAvgPr() + "," +
                                    avMINAS.getAvgRe() + "," +
                                    avMINAS.getAvgSA() + "," +
                                    avMINAS.getQtdeNP() + "," +
                                    avMINAS.getUnk().stream().mapToInt(p -> p).sum() + "," +
                                    avMINAS.getRemovedUnk().stream().mapToInt(p -> p).sum() + "\n");
                            fileResults.close();
                        }
                    }
                }
            }
        }
    }

    public static EvaluatorBR MINAS_BR(ArrayList<Instance> train,
            ArrayList<Instance> test,
            int L,
            double k_ini,
            int theta,
            int omega,
            double f,
            String outputDirectory) throws IOException, Exception {

        // Create output files
        FileWriter filePredictions = new FileWriter(new File(outputDirectory + "/faseOnlineInfo_.txt"), false); // Armazena
                                                                                                                // informações
                                                                                                                // da
                                                                                                                // fase
                                                                                                                // online
        // filePredictions.write("timestamp;actual;predicted" + "\n");
        FileWriter fileOff = new FileWriter(new File(outputDirectory + "/faseOfflineInfo.txt"), false); // Armazena
                                                                                                        // informações
                                                                                                        // da fase
                                                                                                        // online
        FileWriter fileOut = new FileWriter(new File(outputDirectory + "/results.txt"), false); // Armazena informações
                                                                                                // da fase de
                                                                                                // treinamento

        int evaluationWindowSize = (int) Math.ceil(test.size() / 50);

        OfflinePhase treino = new OfflinePhase(train, k_ini, fileOff, outputDirectory);
        Model model = treino.getModel();
        model.setEvaluationWindowSize(evaluationWindowSize);
        model.writeCurrentCardinality(1, outputDirectory);

        fileOff.write("Known Classes: " + model.getAllLabel().size() + "\n");
        fileOff.write("Train label cardinality: " + model.getCurrentCardinality() + "\n");
        // fileOff.write("Windows label cardinality: " +
        // Arrays.toString(windowsCardinalities) + "\n");
        fileOff.write("Number of examples: " + (train.size() + test.size()) + "\n");
        fileOff.write("Number of attributes: " + train.get(0).numInputAttributes() + "\n");

        EvaluatorBR av = new EvaluatorBR(L, model.getModel().keySet(), "MINAS-BR");
        OnlinePhase onlinePhase = new OnlinePhase(theta, f, outputDirectory, fileOut, "kmeans+leader");

        // Classification phase
        for (int i = 0; i < test.size(); i++) {
            onlinePhase.incrementarTimeStamp();
            System.out.println("Timestamp: " + onlinePhase.getTimestamp());
            onlinePhase.classify(model, av, test.get(i), filePredictions);

            // for each model deletes the micro-clusters wich have not been used
            if ((onlinePhase.getTimestamp() % omega) == 0) {
                model.resetMtxLabelFrequencies(omega);
                model.updateCardinality(omega);
                model.writeCurrentCardinality(onlinePhase.getTimestamp(), outputDirectory);
                model.writeBayesRulesElements(onlinePhase.getTimestamp(), outputDirectory);
                model.clearSortTimeMemory(omega, onlinePhase.getTimestamp(), fileOut, false);
                onlinePhase.removeOldMicroClusters(omega, model, fileOut);
            }
            if ((onlinePhase.getTimestamp() % evaluationWindowSize) == 0) {
                // model.writeBayesRulesElements(onlinePhase.getTimestamp(), outputDirectory);
                // model.writeCurrentCardinality(onlinePhase.getTimestamp(), outputDirectory);
                model.associatesNPs(evaluationWindowSize, onlinePhase.getTimestamp(), "JI");
                // av.getDeletedExamples().add(model.getShortTimeMemory().getQtdeExDeleted());
                av.updateExampleBasedMeasure(model, evaluationWindowSize);
                av.updateLabelBasedMeasure(model, evaluationWindowSize);
            }
            if (i == test.size() - 1 && (onlinePhase.getTimestamp() % evaluationWindowSize) > 0) {
                model.associatesNPs(evaluationWindowSize, onlinePhase.getTimestamp(), "JI");
                av.updateExampleBasedMeasure(model, evaluationWindowSize);
                av.updateLabelBasedMeasure(model, evaluationWindowSize);
            }
        }
        onlinePhase.getExtInfo().close();
        av.setQtdeNP(model.getNPs().size());
        av.writeMeasuresOverTime(outputDirectory);
        av.writeConceptEvolutionNP(model, outputDirectory);
        fileOut.close();
        filePredictions.close();
        fileOff.close();
        System.out.println("Number of examples sent to short-time-memory = " + onlinePhase.getExShortTimeMem());
        model.getPnInfo()
                .write("Number of examples sent to short-time-memory = " + onlinePhase.getExShortTimeMem() + "\n");
        System.out.println(
                "Number of examples removed from short-time-memory = " + model.getShortTimeMemory().getQtdeExDeleted());
        model.getPnInfo().write("Number of examples removed from short-time-memory = "
                + model.getShortTimeMemory().getQtdeExDeleted() + "\n");
        System.out.println("Number of NPs = " + model.getNPs().size());
        model.getPnInfo().write("Number of NPs = " + model.getNPs().size() + "\n");
        model.getPnInfo().close();
        return av;
    }

    /**
     * Experiments to compare MINAS-BR with others methods
     *
     * @param dataSetName
     * @param dataSetPath
     * @param omega           window size
     * @param theta           limit of short-term memory
     * @param f
     * @param algOn
     * @param evMetric
     * @param outputDirectory
     * @throws Exception
     */
    public static void experimentsMethods(String trainPath,
            String testPath,
            String outputDirectory,
            int L,
            double k_ini,
            String theta,
            String omega,
            String f,
            String algOn,
            String evMetric) throws Exception {

        FilesOutput.createDirectory(outputDirectory);
        ArrayList<Instance> train = new ArrayList<Instance>();
        ArrayList<Instance> test = new ArrayList<Instance>();

        MultiTargetArffFileStream file = new MultiTargetArffFileStream(trainPath, String.valueOf(L));
        file.prepareForUse();
        while (file.hasMoreInstances()) {
            train.add(file.nextInstance().getData());
        }
        file.restart();

        file = new MultiTargetArffFileStream(testPath, String.valueOf(L));
        file.prepareForUse();

        while (file.hasMoreInstances()) {
            test.add(file.nextInstance().getData());
        }
        file.restart();

        ArrayList<Instance> aux = new ArrayList<>();
        aux.addAll(train);
        aux.addAll(test);
        float cardinalityTrain = DataSetUtils.getCardinality(train, L);
        float labelCardinality = DataSetUtils.getCardinality(aux, L);
        FileWriter DsInfos = new FileWriter(new File(outputDirectory + "/dataSetInfo.txt"), false);

        DsInfos.write("Train label cardinality: " + cardinalityTrain + "\n");
        DsInfos.write("General label cardinality: " + labelCardinality + "\n");
        // DsInfos.write("Windows label cardinality: " +
        // Arrays.toString(windowsCardinalities) + "\n");
        DsInfos.write("Number of examples: " + train.size() + test.size() + "\n");
        DsInfos.write("Number of attributes: " + train.get(0).numInputAttributes() + "\n");
        DsInfos.close();

        ArrayList<Evaluator> av = new ArrayList<Evaluator>();
        av.add(MINAS_BR(train,
                test,
                L,
                k_ini,
                Integer.valueOf(theta),
                Integer.valueOf(omega),
                Double.valueOf(f),
                outputDirectory));

        EvaluatorBR.writesAvgResults(av, outputDirectory);
        Evaluator.writeMeasuresOverTime(av, outputDirectory);
    }

    private static void removeClasses(String dataSetPath) throws IOException {
        MultiTargetArffFileStream stream = new MultiTargetArffFileStream(dataSetPath, "1");
        stream.prepareForUse();

        FileWriter dataSetFile = new FileWriter(new File(dataSetPath.replace(".arff", "-CE.arff")), false);

        // write the file's header
        dataSetFile.write("@relation \n");
        dataSetFile.write("@attribute att1 numeric \n");
        dataSetFile.write("@attribute att2 numeric \n");
        dataSetFile.write("@attribute class \n");
        dataSetFile.write("\n");
        dataSetFile.write("@data\n");

        int cont1 = 1;
        int cont2 = 1;
        while (cont1 <= 50000) {
            Instance inst = stream.nextInstance().getData();
            System.out.println(inst.value(2));
            if (inst.value(2) != 1.0 && inst.value(2) != 2.0) {
                for (int i = 0; i < inst.numAttributes(); i++) {
                    dataSetFile.write(inst.value(i) + ",");
                }
                dataSetFile.write("\n");
            }
            cont1++;
            cont2++;
        }
        while (cont2 <= 100000) {
            Instance inst = stream.nextInstance().getData();
            System.out.println(inst.value(2));
            if (inst.value(2) != 2) {
                for (int i = 0; i < inst.numAttributes(); i++) {
                    dataSetFile.write(inst.value(i) + ",");
                }
                dataSetFile.write("\n");
            }
            cont1++;
            cont2++;
        }
        while (stream.hasMoreInstances()) {
            Instance inst = stream.nextInstance().getData();
            for (int i = 0; i < inst.numAttributes(); i++) {
                dataSetFile.write(inst.value(i) + ",");
            }
            dataSetFile.write("\n");
        }
        dataSetFile.close();
    }

}
