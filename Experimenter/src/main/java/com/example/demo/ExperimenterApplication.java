package com.example.demo;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;

import javax.sound.midi.Soundbank;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.meta.*;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.TimingUtils;
import moa.streams.ArffFileStream;
import moa.streams.generators.AgrawalGenerator;
import moa.streams.generators.HyperplaneGenerator;
import moa.streams.generators.LEDGenerator;
import moa.streams.generators.RandomRBFGenerator;
import moa.streams.generators.RandomTreeGenerator;

@SpringBootApplication
public class ExperimenterApplication {

	public static void main(String[] args) throws IOException {
		SpringApplication.run(ExperimenterApplication.class, args);
		Experiment();
	}

	public static void print_array(int mat[], BufferedWriter writer) throws IOException {
		// Loop through all elements
		for (int i = 0; i < mat.length; i++)
			writer.write(mat[i] + " ");
	}
	
	public static void print2D(boolean mat[][], BufferedWriter writer) throws IOException  
    { 
        // Loop through all rows 
        for (int i = 0; i < mat.length; i++) {
  
            // Loop through all elements of current row 
            for (int j = 0; j < mat[i].length; j++) {
            	writer.write(Boolean.toString(mat[i][j]) + " ");
            }
            writer.newLine();    
        }
        	
    } 
	
// test m
	public static void Experiment() throws IOException {
		
		String dirPath = new String("D:Leiden/AML/data/");
		File folder = new File(dirPath+"arff/");
		File[] listOfFiles = folder.listFiles();

		for(int i=0; i<listOfFiles.length; i++) {
			
			System.out.println("Processing file: "+ listOfFiles[i]);
			
			BufferedWriter writer = new BufferedWriter(new FileWriter(dirPath+"/results/"+listOfFiles[i].getName()));

			int numInstances = 10000;
			boolean isTesting = true;

			Classifier learner = new HeterogeneousEnsembleBlast();
			//HyperplaneGenerator stream = new HyperplaneGenerator();

			ArffFileStream stream = new ArffFileStream(dirPath+"/arff/"+listOfFiles[i].getName(), -1);
			
			stream.prepareForUse();

			learner.prepareForUse();
			learner.setModelContext(stream.getHeader());

			int numberSamplesCorrect = 0;
			int numberSamples = 0;
			boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
			long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();

			while (stream.hasMoreInstances() && numberSamples < numInstances) {
				Instance trainInst = stream.nextInstance().getData();
				if (isTesting) {
					if (learner.correctlyClassifies(trainInst)) {
						numberSamplesCorrect++;
					}
				}
				numberSamples++;
				learner.trainOnInstance(trainInst);
				// System.out.println(numberSamples);
			}
			double accuracy = 100.0 * (double) numberSamplesCorrect / (double) numberSamples;
			double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);
			
			writer.write(
					numberSamples + " instances processed with " + accuracy + "% accuracy in " + time + " seconds.");

			writer.newLine();
			
			int[] activeClassifierWins = ((HeterogeneousEnsembleBlast) learner).getActiveClassifierWins();    
			
			for (int j = 0; j < ((HeterogeneousEnsembleBlast) learner).ensemble.length; j++) {
				writer.write(Integer.toString(i));
				writer.newLine();
				writer.write(((HeterogeneousEnsembleBlast) learner).ensemble[i].getModel().getClass().toString());
				writer.newLine();
			}

			print_array(activeClassifierWins, writer);
			
			writer.newLine();
			
			final boolean[][] onlineHistory = ((HeterogeneousEnsembleBlast) learner).getOnlineHistory();
			//System.out.println(Arrays.deepToString(onlineHistory));
			writer.newLine();
			print2D(onlineHistory,writer);
			
			writer.close();
		}
		
	}

}
