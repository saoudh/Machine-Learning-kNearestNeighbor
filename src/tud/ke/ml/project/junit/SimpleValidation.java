package tud.ke.ml.project.junit;

import static org.junit.Assert.assertNotNull;

import java.io.File;
import java.util.LinkedList;
import java.util.List;

import org.junit.BeforeClass;
import org.junit.Test;

import tud.ke.ml.project.classifier.AbstractNearestNeighbor;
import tud.ke.ml.project.classifier.NearestNeighbor;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KENearestNeighbor;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;
import weka.core.converters.ArffLoader;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class SimpleValidation {
	private static List<Instances> data;
	private static RemovePercentage filterTrain, filterTest;
	private static KENearestNeighbor keClassifier = new KENearestNeighbor();
	private static IBk wekaClassifier = new IBk();

	/**
	 * This test validates if getMatrikelNumbers has been implemented.
	 */
	@Test
	public void testGroupNumber() throws Exception {
		AbstractNearestNeighbor classifier = new NearestNeighbor();
		assertNotNull(classifier.getMatrikelNumbers());
	}

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		data = new LinkedList<Instances>();
		ArffLoader loader = new ArffLoader();
		Instances instances;

		loader.setFile(new File("data/contact-lenses.arff"));
		// loader.setFile(new File("data/credit-g.arff")); // using credit-g.arff leads to one not passed test
		instances = loader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);
		data.add(instances);

		keClassifier = new KENearestNeighbor();
		wekaClassifier = new IBk();

		filterTrain = new RemovePercentage();
		filterTrain.setPercentage(AdvancedValidation.testSplitPercentage);
		filterTest = new RemovePercentage();
		filterTest.setPercentage(AdvancedValidation.testSplitPercentage);
		filterTest.setInvertSelection(true);
	}

	/**
	 * This test validates if the model is getting learned without throwing exceptions.
	 */
	@Test
	public void testLearnModel() throws Exception {
		for (Instances instances : data) {
			keClassifier.buildClassifier(instances);
		}
	}

	/**
	 * This test validates if the classifier is able to classify new instances without throwing exceptions.
	 */
	@Test
	public void testClassify() throws Exception {
		keClassifier.setkNearest(2);
		keClassifier.setMetric(new SelectedTag(1, KENearestNeighbor.TAGS_DISTANCE));
		keClassifier.setDistanceWeighting(new SelectedTag(1, KENearestNeighbor.TAGS_WEIGHTING));

		for (Instances instances : data) {
			filterTrain.setInputFormat(instances);
			filterTest.setInputFormat(instances);
			keClassifier.buildClassifier(Filter.useFilter(instances, filterTrain));
			for (Instance instance : Filter.useFilter(instances, filterTest)) {
				keClassifier.classifyInstance(instance);
			}
		}

		keClassifier.setkNearest(10);
		keClassifier.setMetric(new SelectedTag(1, KENearestNeighbor.TAGS_DISTANCE));
		keClassifier.setDistanceWeighting(new SelectedTag(1, KENearestNeighbor.TAGS_WEIGHTING));

		for (Instances instances : data) {
			filterTrain.setInputFormat(instances);
			filterTest.setInputFormat(instances);
			keClassifier.buildClassifier(Filter.useFilter(instances, filterTrain));
			for (Instance instance : Filter.useFilter(instances, filterTest)) {
				keClassifier.classifyInstance(instance);
			}
		}
	}

	/**
	 * This test the correctness of the unweighted Manhattan distance implementation
	 */
	@Test
	public void testCorrectnessUnweightedManhattank1() throws Exception {
		keClassifier.setkNearest(1);
		keClassifier.setMetric(new SelectedTag(0, KENearestNeighbor.TAGS_DISTANCE));
		keClassifier.setDistanceWeighting(new SelectedTag(0, KENearestNeighbor.TAGS_WEIGHTING));

		wekaClassifier.setKNN(1);
		NearestNeighbourSearch search = new LinearNNSearch();
		ManhattanDistance df = new ManhattanDistance();
		df.setDontNormalize(true);
		search.setDistanceFunction(df);
		search.setMeasurePerformance(false);
		wekaClassifier.setNearestNeighbourSearchAlgorithm(search);
		wekaClassifier.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING));

		for (Instances instances : data) {
			filterTrain.setInputFormat(instances);
			filterTest.setInputFormat(instances);
			Instances train = Filter.useFilter(instances, filterTrain);
			keClassifier.buildClassifier(train);
			wekaClassifier.buildClassifier(train);
			Instances test = Filter.useFilter(instances, filterTest);
			for (Instance instance : test) {
				AdvancedValidation.comparePredictions(keClassifier, wekaClassifier, instance);
			}
		}
	}

	/**
	 * This test validates the correctness of a higher k (10) classification
	 */
	@Test
	public void testCorrectnessUnweightedEuclideank1() throws Exception {
		keClassifier.setkNearest(1);
		keClassifier.setMetric(new SelectedTag(1, KENearestNeighbor.TAGS_DISTANCE));
		keClassifier.setDistanceWeighting(new SelectedTag(0, KENearestNeighbor.TAGS_WEIGHTING));

		wekaClassifier.setKNN(1);
		NearestNeighbourSearch search = new LinearNNSearch();
		EuclideanDistance df = new EuclideanDistance();
		df.setDontNormalize(true);
		search.setDistanceFunction(df);
		search.setMeasurePerformance(false);
		wekaClassifier.setNearestNeighbourSearchAlgorithm(search);
		wekaClassifier.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING));

		for (Instances instances : data) {
			filterTrain.setInputFormat(instances);
			filterTest.setInputFormat(instances);
			Instances train = Filter.useFilter(instances, filterTrain);
			keClassifier.buildClassifier(train);
			wekaClassifier.buildClassifier(train);
			Instances test = Filter.useFilter(instances, filterTest);
			for (Instance instance : test) {
				AdvancedValidation.comparePredictions(keClassifier, wekaClassifier, instance);
			}
		}
	}

	/**
	 * This test validates the correctness of a higher k (10) classification
	 */
	@Test
	public void testCorrectnessWeightedManhattank10() throws Exception {
		keClassifier.setkNearest(10);
		keClassifier.setMetric(new SelectedTag(0, KENearestNeighbor.TAGS_DISTANCE));
		keClassifier.setDistanceWeighting(new SelectedTag(1, KENearestNeighbor.TAGS_WEIGHTING));

		wekaClassifier.setKNN(10);
		NearestNeighbourSearch search = new LinearNNSearch();
		ManhattanDistance df = new ManhattanDistance();
		df.setDontNormalize(true);
		search.setDistanceFunction(df);
		search.setMeasurePerformance(false);
		wekaClassifier.setNearestNeighbourSearchAlgorithm(search);
		wekaClassifier.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING));

		for (Instances instances : data) {
			filterTrain.setInputFormat(instances);
			filterTest.setInputFormat(instances);
			Instances train = Filter.useFilter(instances, filterTrain);
			keClassifier.buildClassifier(train);
			wekaClassifier.buildClassifier(train);
			Instances test = Filter.useFilter(instances, filterTest);
			for (Instance instance : test) {
				AdvancedValidation.comparePredictions(keClassifier, wekaClassifier, instance);
			}
		}
	}

	/**
	 * This tests validates the inverse weighted, euclidean distance metric.
	 */
	@Test
	public void testCorrectnessWeightedEuclideank10() throws Exception {
		NearestNeighbourSearch search = new LinearNNSearch();

		keClassifier.setkNearest(10);
		keClassifier.setMetric(new SelectedTag(1, KENearestNeighbor.TAGS_DISTANCE));
		keClassifier.setDistanceWeighting(new SelectedTag(1, KENearestNeighbor.TAGS_WEIGHTING));

		wekaClassifier.setKNN(10);
		search = new LinearNNSearch();
		EuclideanDistance df = new EuclideanDistance();
		df.setDontNormalize(true);
		search.setDistanceFunction(df);
		search.setMeasurePerformance(false);
		wekaClassifier.setNearestNeighbourSearchAlgorithm(search);
		wekaClassifier.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING));

		for (Instances instances : data) {
			filterTrain.setInputFormat(instances);
			filterTest.setInputFormat(instances);
			Instances train = Filter.useFilter(instances, filterTrain);
			keClassifier.buildClassifier(train);
			wekaClassifier.buildClassifier(train);
			Instances test = Filter.useFilter(instances, filterTest);
			for (Instance instance : test) {
				AdvancedValidation.comparePredictions(keClassifier, wekaClassifier, instance);
			}
		}
	}
}
