/**
 *  Big Data Technology
 *
 *  Created on: July 27, 2019
 *  Data Scientist: Tung Dang
 */

package jp.ac.utokyo.biobigdata.core.sequence;

import jp.ac.utokyo.biobigdata.core.util.Equals;
import jp.ac.utokyo.biobigdata.core.util.Hashcoder;

/**
 * Used in Sequence as the unique indentifier.
 * Set the DataSource input to know the source of the id. 
 * This allows a SequenceProxy to gather features or related sequences Protein->Gene 
 */

public class AccessionID {
    private String id = null;
    private DataSource source = DataSource.LOCAL;
    private Integer version;
    private String identifier = null;

    public AccessionID() {
        id = "";
    }

    public AccessionID(String id) {
        this.id = id.trim();
        this.source = DataSource.LOCAL;
    }

    public AccessionID(String id, DataSource source) {
        this.id = id.trim();
        this.source = source;
    }

    public AccessionID(String id, DataSource source, Integer version, String identifier) {
        this.id = id;
        this.source = source;
        this.version = version;
        this.identifier = identifier;
    }

    public String getID() {
        return id;
    }

    public DataSource getDataSource() {
        return source;
    }

    public Integer getVersion() {
        return version;
    }

    public void setVersion(Integer version) {
        this.version = version;
    }

    public String getIdentifier() {
        return identifier;
    }

    public void setIdentifier(String identifier) {
        this.identifier = identifier;
    }

    @Override
    public boolean equals(Object o) {
        boolean equals = false;
        if (Equals.classEqual(this, o)) {
            AccessionID i = (AccessionID) o;
            equals = (Equals.equal(getID(), i.getID())
                            && Equals.equal(getDataSource(), i.getDataSource())
                            && Equals.equal(getIdentifier(), i.getIdentifier())
                            && Equals.equal(getVersion(), i.getVersion()));
        }
        return equals;
    }

    @Override
    public int hashCode() {
        int r = Hashcoder.SEED;
        r = Hashcoder.hash(r, getID());
        r = Hashcoder.hash(r, getDataSource());
        r = Hashcoder.hash(r, getIdentifier());
        r = Hashcoder.hash(r, getVersion());

        return r;
    }
}