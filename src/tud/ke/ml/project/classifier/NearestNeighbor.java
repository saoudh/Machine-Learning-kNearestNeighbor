package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import tud.ke.ml.project.util.Pair;

/**
 * This implementation assumes the class attribute is always available (but probably not set).
 */
public class NearestNeighbor extends AbstractNearestNeighbor implements Serializable {
	private static final long serialVersionUID = 8662234558169046563L;

	protected double[] scaling;
	protected double[] translation;

	@Override
	public String getMatrikelNumbers() {
		throw new NotImplementedException();
	}

	@Override
	protected void learnModel(List<List<Object>> data) {
		throw new NotImplementedException();
	}

	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		throw new NotImplementedException();
	}

	@Override
	protected Map<Object, Double> getUnweightedVotes(List<Pair<List<Object>, Double>> subset) {
		throw new NotImplementedException();
	}

	@Override
	protected Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset) {
		throw new NotImplementedException();
	}

	@Override
	protected Object getWinner(Map<Object, Double> votes) {
		throw new NotImplementedException();
	}

	@Override
	protected List<Pair<List<Object>, Double>> getKNearest(List<Object> data) {
		throw new NotImplementedException();
	}

	@Override
	protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {
		throw new NotImplementedException();
	}

	@Override
	protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {
		throw new NotImplementedException();
	}

	@Override
	protected double[][] normalizationScaling() {
		throw new NotImplementedException();
	}
}
