package org.mai.dep110.stream.iris;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.ToDoubleFunction;

public class IrisDataSetHelper {

    private List<Iris> dataSet;

    public IrisDataSetHelper(List<Iris> dataSet) {
        this.dataSet = dataSet;
    }

    public Double getAverage(ToDoubleFunction<Iris> func) {
        throw new NotImplementedException();
    }

    public List<Iris> filter(Predicate<Iris> predicate) {
        throw new NotImplementedException();
    }

    public Double getAverageWithFilter(Predicate<Iris> filter, ToDoubleFunction<Iris> mapFunction) {
        throw new NotImplementedException();
    }

    public Map groupBy(Function groupFunction) {
        throw new NotImplementedException();
    }

    public Map maxFromGroupedBy(Function groupFunction, ToDoubleFunction<Iris> obtainMaximisationValueFunction) {
        throw new NotImplementedException();
    }
}
