/**
 *  Big Data Technology
 *
 *  Created on: July 27, 2019
 *  Data Scientist: Tung Dang
 */

package jp.ac.utokyo.biobigdata.core.sequence.template;

import java.util.List;
import java.util.Set;

public interface CompoundSet <C extends Compound> {
    public int getMaxSingleCompoundStringLength();
    public boolean isCompoundStringLengthEqual();
    public C getCompoundForString(String string);
    public String getStringForCompound(C compound);
    public boolean compoundEquivalent(C compoundOne, C compoundTwo);
    public boolean isValidSequence(Sequence<C> sequence);
    public Set<C> getEquivalentCompounds(C compound);
    public boolean hasCompound(C compound);
    public List<C> getAllCompounds();
    boolean isComplementable();
}