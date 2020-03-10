package com.taobao.android.utils;


public class ShiftList<T> {
    private Node<T> root;
    private Node<T> last;
    private int maxCapacity;
    private int currentSize;

    public ShiftList(int maxCapacity) {
        this.maxCapacity = maxCapacity;
        this.root = null;
        this.last = null;
    }

    public void add(T data) {
        Node<T> node = new Node<T>(data);

        if (root == null) {
            root = node;
            last = node;
        } else {
            last.setNext(node);
            last = node;

            // Move root forward
            if (currentSize == maxCapacity) {
                root = root.getNext();
            }
        }

        // Increment only until full capacity
        if (currentSize < maxCapacity)
            currentSize++;
    }

    public T get(int index) {
        Node<T> next = root;
        for (int i = 0; i < index; i++) {
            next = next.getNext();
        }
        return next.getData();
    }

    public int getSize() {
        return currentSize;
    }
}

