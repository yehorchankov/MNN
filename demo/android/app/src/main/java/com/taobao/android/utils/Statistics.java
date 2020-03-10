package com.taobao.android.utils;

public class Statistics {
    public static double mean(ShiftList<Double> list) {
        double sum = 0;
        for (int i = 0; i < list.getSize(); i++) {
            sum += list.get(i);
        }
        return sum / list.getSize();
    }
}
