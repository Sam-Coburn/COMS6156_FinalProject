// 
// Decompiled by Procyon v0.5.29
// 

class SLinkedList
{
    private int data;
    private SLinkedList next;
    
    public SLinkedList() {
        this.data = 0;
        this.next = null;
    }
    
    public SLinkedList(final int data) {
        this.data = data;
        this.next = null;
    }
    
    public SLinkedList InsertNext(final int n) {
        final SLinkedList list = new SLinkedList(n);
        if (this.next != null) {
            list.next = this.next;
            this.next = list;
        }
        else {
            list.next = null;
            this.next = list;
        }
        return list;
    }
    
    public int DeleteNext() {
        if (this.next == null) {
            return 0;
        }
        final SLinkedList next = this.next;
        this.next = this.next.next;
        return 1;
    }
    
    public void Traverse(SLinkedList next) {
        if (next == null) {
            next = this;
        }
        System.out.println("\n\nTraversing in Forward Direction\n\n");
        while (next != null) {
            System.out.println(next.data);
            next = next.next;
        }
    }
    
    public SLinkedList Reverse_With_Bubble(final SLinkedList list) {
        SLinkedList list2 = null;
        if (list == null) {
            return null;
        }
        while (list.next != list2) {
            SLinkedList next = list;
            do {
                final int data = next.data;
                next.data = next.next.data;
                next.next.data = data;
                next = next.next;
            } while (next.next != list2);
            list2 = next;
        }
        return list;
    }
    
    public SLinkedList Reverse(SLinkedList list) {
        if (list == null) {
            return null;
        }
        SLinkedList next = null;
        while (list != null) {
            final SLinkedList next2 = list.next;
            list.next = next;
            list = (next = next2);
        }
        return next;
    }
    
    public SLinkedList Reverse_with_copy(SLinkedList next) {
        SLinkedList next2 = null;
        while (next != null) {
            final SLinkedList list = new SLinkedList(next.data);
            list.next = next2;
            next = next.next;
            next2 = list;
        }
        return next2;
    }
    
    public static void main(final String[] array) {
        final SLinkedList list = new SLinkedList(1);
        list.InsertNext(2).InsertNext(3).InsertNext(4).InsertNext(5).InsertNext(6).InsertNext(7).InsertNext(8);
        list.Traverse(null);
        list.Reverse(list).Traverse(null);
    }
}
