/**
 *  Big Data Technology
 *
 *  Created on: July 27, 2019
 *  Data Scientist: Tung Dang
 */

package jp.ac.utokyo.biobigdata.core.sequence.template;

import jp.ac.utokyo.biobigdata.core.sequence.AccessionID;

public interface Accessioned {
    AccessionID getAccession();
}