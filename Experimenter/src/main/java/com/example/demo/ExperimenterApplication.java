package com.example.demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.meta.*;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.TimingUtils;
import moa.streams.generators.AgrawalGenerator;
import moa.streams.generators.HyperplaneGenerator;
import moa.streams.generators.LEDGenerator;
import moa.streams.generators.RandomRBFGenerator;
import moa.streams.generators.RandomTreeGenerator;

@SpringBootApplication
public class ExperimenterApplication {

	public static void main(String[] args) {
		SpringApplication.run(ExperimenterApplication.class, args);
		Experiment();
	}

	public static void print_array(int mat[]) {
		// Loop through all elements
		for (int i = 0; i < mat.length; i++)
			System.out.print(mat[i] + " ");
	}
// test m
	// rtetst Abhi
	public static void Experiment() {

		int numInstances = 10000;
		boolean isTesting = true;

		Classifier learner = new HeterogeneousEnsembleBlast();
		HyperplaneGenerator stream = new HyperplaneGenerator();

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
		System.out.println(
				numberSamples + " instances processed with " + accuracy + "% accuracy in " + time + " seconds.");

		int[] activeClassifierWins = ((HeterogeneousEnsembleBlast) learner).getActiveClassifierWins();

		for (int i = 0; i < ((HeterogeneousEnsembleBlast) learner).ensemble.length; i++) {
			System.out.println(i);
			System.out.println(((HeterogeneousEnsembleBlast) learner).ensemble[i].getModel().getClass());
		}

		print_array(activeClassifierWins);
	}

}
