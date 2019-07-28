/**
 *  Big Data Technology
 *
 *  Created on: July 27, 2019
 *  Data Scientist: Tung Dang
 */

package jp.ac.utokyo.biobigdata.core.sequence.template;

public interface Compound {
    public boolean equalsIgnoreCase(Compound compound);
    public String getDescription();
    public void setDescription(String description);
    public String getShortName();
    public void setShortName(String shortName);
    public String getLongName();
    public void setLongName(String longName);
    public Float getMolecularWeight();
    public void setMolecularWeight(Float molecularWeight);
}
