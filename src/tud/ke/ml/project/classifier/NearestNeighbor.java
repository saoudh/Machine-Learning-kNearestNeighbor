package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import tud.ke.ml.project.util.Pair;

/**
 * This implementation assumes the class attribute is always available (but
 * probably not set).
 */
public class NearestNeighbor extends AbstractNearestNeighbor implements Serializable {
	private static final long serialVersionUID = 8662234558169046563L;

	protected double[] scaling;
	protected double[] translation;
	protected List<List<Object>> data;

	@Override
	public String getMatrikelNumbers() {
		return "TODO: matriculationnumber";
	}

	@Override
	protected void learnModel(List<List<Object>> data) {
		this.data = data;
	}

	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		// every class with it's distance as double, e.g. "yes"=2.4, "no"=4.5
		Map<Object, Double> votes = getUnweightedVotes(subset);

		// TODO: return winning class with most votes

		return getWinner(votes);
	}

	@Override
	protected Map<Object, Double> getUnweightedVotes(List<Pair<List<Object>, Double>> subset) {

		for (Pair pair : subset) {
			System.out.println("getUnweightedVotes=" + pair.toString());
		}
		return null;
	}

	@Override
	protected Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset) {
		return null;
	}

	@Override
	protected Object getWinner(Map<Object, Double> votes) {
		return null;
	}

	@Override
	protected List<Pair<List<Object>, Double>> getKNearest(List<Object> data) {
		// raw data is passed and a list of pairs of instances with their distances is
		// returned
		List<Pair<List<Object>, Double>> listOfInstanceDistancePairs = new ArrayList<Pair<List<Object>, Double>>();

		// loop over all instances in the training data and determine the distance to the passed instance
		for (List<Object> instance2 : this.data) {
			double distance = determineManhattanDistance(data, instance2);
			listOfInstanceDistancePairs.add(new Pair(instance2, distance));
		}

		// TODO: return only k instances: sort by lowest distance and return k elements
		
		// return list of k pairs of the instances in the training data with their distance to the passed instance
		return listOfInstanceDistancePairs;
	}

	@Override
	protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {
		double distance = 0.0;
		// excluding the class-attribute
		for (int i = 0; i < getClassAttributeIndex() - 1; i++) {

			// attribute i
			Object attrOfInstance1 = instance1.get(i);
			Object attrOfInstance2 = instance2.get(i);
			// if attribute i of instance 1 equals that of instance 2 than distance is 0 else 1
			if (!attrOfInstance1.toString().equals(attrOfInstance2.toString())) 
				distance += 1;
			
		}
		return distance;
	}

	@Override
	protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {

		return 0.00;
	}

	@Override
	protected double[][] normalizationScaling() {
		return new double[][] { { 0.0 } };
	}
}
