/**
 *  Big Data Technology
 *
 *  Created on: July 27, 2019
 *  Data Scientist: Tung Dang
 */

package jp.ac.utokyo.biobigdata.core.util;

import java.lang.reflect.Array;

public class Hashcoder{
    public static final int SEED = 9;
    public static final int PRIME = 79;

    public static int hash(int seed, boolean b) {
        return (PRIME * seed) + (b ? 1 : 0);
    }

    public static int hash(int seed, char c) {
        return (PRIME * seed) + c;
    }

    public static int hash(int seed, int i) {
        return (PRIME * seed) + i;
    }

    public static int hash(int seed, long l) {
        return (PRIME * seed) + (int) (l ^ (l >>> 32));
    }

    public static int hash(int seed, float f) {
        return hash(seed, Float.floatToIntBits(f));
    }

    public static int hash(int seed, double d) {
        return hash(seed, Double.doubleToLongBits(d));
    }

    public static int hash(int seed, Object o) {
        int result = seed;
        if (o == null) {
            result = hash(result, 0);
        }
        else if (!o.getClass().isArray()) {
            result = hash(result, o.hashCode());
        }
        else {
            int length = Array.getLength(o);
            for (int i=0; i < length; i++)
            {
                Object item = Array.get(o, i);
                result = hash(result, item);
            }
        }
        return result;
    }
}