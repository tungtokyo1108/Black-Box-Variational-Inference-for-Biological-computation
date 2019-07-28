/**
 *  Big Data Technology
 *
 *  Created on: July 28, 2019
 *  Data Scientist: Tung Dang
 */

package jp.ac.utokyo.biobigdata.core.sequence.template;

public interface SequenceView<C extends Compound> extends Sequence<C> {
    public Sequence<C> getViewedSequence();
    public Integer getBioStart();
    public Integer getBioEnd();
}