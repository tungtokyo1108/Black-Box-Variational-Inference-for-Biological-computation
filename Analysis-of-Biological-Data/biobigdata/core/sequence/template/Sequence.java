/**
 *  Big Data Technology
 *
 *  Created on: July 27, 2019
 *  Data Scientist: Tung Dang
 */

package jp.ac.utokyo.biobigdata.core.sequence.template;

import java.util.List;

public interface Sequence<C extends Compound> extends Iterable<C>, Accessioned {
    public int getLength();
    public C getCompoundAt(int position);
    
    // Scans through the Sequence looking for the first occurance of the given compound
    public int getIndexOf(C compound);

    // Scans through the Sequenc looking for the last occurance of the given compound
    public int getLastIndexOf(C compound);

    // Return the String representation of the Sequence
    public String getSequenceAsString();

    // Return the Sequence as a List of compounds
    public List<C> getAsList();

    public SequenceView<C> getSubSequence(Integer start, Integer end);

    public CompoundSet<C> getCompoundSet();

    public int countCompounds(C... compounds);

    public SequenceView<C> getInverse();

}