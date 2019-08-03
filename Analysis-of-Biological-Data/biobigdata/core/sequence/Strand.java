/**
 *  Big Data Technology
 *
 *  Created on: July 28, 2019
 *  Data Scientist: Tung Dang
 */

package jp.ac.utokyo.biobigdata.core.sequence;

// Provides a way to represent the strand of a sequence, locations, hit or feature

public enum Strand {
    POSITIVE("+", 1), NEGATIVE("-", -1), UNDEFINED(".", 0);
    private final String stringRepresentation;
    private final int numericRepresentation;

    private Strand(String stringRepresentation, int numericRepresentation) {
        this.stringRepresentation = stringRepresentation;
        this.numericRepresentation = numericRepresentation;
    }

    public int getNumericRepresentation() {
        return numericRepresentation;
    }

    public String getStringRepresentation() {
        return stringRepresentation;
    }

    public Strand getReverse() {
        switch (this) {
            case POSITIVE:
                return NEGATIVE;
            case NEGATIVE:
                return POSITIVE;
            default:
                return UNDEFINED;
        }
    }
}
