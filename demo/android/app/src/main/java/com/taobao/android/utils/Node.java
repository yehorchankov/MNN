package com.taobao.android.utils;

class Node<T> {
    private Node<T> next;
    private T data;

    Node (T data) {
        this.data = data;
    }

    void setNext(Node<T> next) {
        this.next = next;
    }

    Node<T> getNext() {
        return next;
    }

    T getData() {
        return data;
    }
}