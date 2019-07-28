/**
 *  Big Data Technology
 *
 *  Created on: July 27, 2019
 *  Data Scientist: Tung Dang
 */

package jp.ac.utokyo.biobigdata.core.util;

public class Equals {
    public static boolean equal(int one, int two) {
        return one == two;
    }

    public static boolean equal(long one, long two) {
        return (one == two);
    }

    public static boolean equal(boolean one, boolean two) {
        return one == two;
    }

    public static boolean equal(Object one, Object two) {
        return one == null && two == null || !(one == null && two == null) && (one == two || one.equals(two));
    }

    public static boolean classEqual(Object one, Object two) {
        return one == two || !(one == null || two == null) && one.getClass() == two.getClass();
    }
}
