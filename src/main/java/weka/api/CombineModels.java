/*
 *  How to use WEKA API in Java 
 *  Copyright (C) 2014 
 *  @author Dr Noureddin M. Sadawi (noureddin.sadawi@gmail.com)
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it as you wish ... 
 *  I ask you only, as a professional courtesy, to cite my name, web page 
 *  and my YouTube Channel!
 *  
 */

package weka.api;
//import required classes
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.lazy.LWL;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Stacking;
import weka.classifiers.meta.Vote;
import weka.classifiers.misc.SerializedClassifier;
import weka.classifiers.trees.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedReader;
import java.io.FileReader;

public class CombineModels {
	public static void main(String[] args) throws Exception {
		//load dataset
		//String data = "C:/Users/vinee/Desktop/defacto_fact2_award_Test.arff";
		//DataSource source = new DataSource(data);

		DataSource source = new DataSource("C:/Users/vinee/Desktop/Test_Mix2.arff");
		//get instances object
		Instances trainingData = source.getDataSet();
		//set class index .. as the last attribute
		if (trainingData.classIndex() == -1) {
			trainingData.setClassIndex(trainingData.numAttributes() - 1);
		}
		trainingData.deleteStringAttributes();


	/*
		*//* Boosting a weak classifier using the Adaboost M1 method
		 * for boosting a nominal class classifier
		 * Tackles only nominal class problems
		 * Improves performance
		 * Sometimes overfits.
		 *//*
		//AdaBoost ..
		AdaBoostM1 m1 = new AdaBoostM1();
		m1.setClassifier(new DecisionStump());//needs one base-classifier
		m1.setNumIterations(20);
		m1.buildClassifier(trainingData);

		*//* Bagging a classifier to reduce variance.
		 * Can do classification and regression (depending on the base model)
		 *//*
		//Bagging ..
		Bagging bagger = new Bagging();
		bagger.setClassifier(new RandomTree());//needs one base-model
		bagger.setNumIterations(25);
		bagger.buildClassifier(trainingData);

		*//*
		 * The Stacking method combines several models
		 * Can do classification or regression.
		 *//*

		*/
		//Stacking ..
		Stacking stacker = new Stacking();
		Classifier smo = (Classifier) weka.core.SerializationHelper.read("data/model/evidence/J48_Model.model");


        Instances test=  new Instances(new BufferedReader(new FileReader("data/eval/defacto_fact2_award_Test.arff")));
        test.setClassIndex(test.numAttributes()-1);
        test.deleteStringAttributes();

        Classifier classifier = (Classifier) weka.core.SerializationHelper.read("data/model/fact/66_33_proof_smo_reg/SMO_Fact_Random_1.model");
        //System.out.println(test.instance(0));
        double[] prediction=classifier.distributionForInstance(test.instance(2));
        System.out.println(prediction[0]);



		Instance newInst = trainingData.instance(848);
		double predSMO[] = smo.distributionForInstance(newInst);

		System.out.println("actualValue"+", "+predSMO[0]);


		//	stacker.setMetaClassifier(new J48());
		stacker.setMetaClassifier(new J48());//needs one meta-model


		Classifier[] classifiers = {
			//	new NaiveBayesMultinomial(),
				new J48(),
				new SMO()

		};
		stacker.setClassifiers(classifiers);//needs one or more models
		stacker.buildClassifier(trainingData);
	/*
		*//*
		 * Class for combining classifiers.
		 * Different combinations of probability estimates for classification are available.
		 *//*
		//Vote ..
		Vote voter = new Vote();
		voter.setClassifiers(classifiers);//needs one or more classifiers
		voter.buildClassifier(trainingData);

		*/


		double postSMO[] = stacker.distributionForInstance(newInst);

		System.out.println("Later Value"+", "+postSMO[0]);
	}
}
