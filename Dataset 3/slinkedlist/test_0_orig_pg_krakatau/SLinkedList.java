// 
// Decompiled by Procyon v0.5.29
// 

class SLinkedList
{
    private int a;
    private SLinkedList b;
    
    public SLinkedList() {
        this.a = 0;
        this.b = null;
    }
    
    private SLinkedList(final int a) {
        this.a = a;
        this.b = null;
    }
    
    private SLinkedList a(final int n) {
        final SLinkedList list = new SLinkedList(n);
        if (this.b != null) {
            list.b = this.b;
            this.b = list;
        }
        else {
            list.b = null;
            this.b = list;
        }
        return list;
    }
    
    private void a(final SLinkedList list) {
        System.out.println("\n\nTraversing in Forward Direction\n\n");
        for (SLinkedList b = this; b != null; b = b.b) {
            System.out.println(b.a);
        }
    }
    
    public static void main(final String[] array) {
        SLinkedList list = new SLinkedList(1);
        list.a(2).a(3).a(4).a(5).a(6).a(7).a(8);
        list.a(null);
        SLinkedList b = null;
        while (list != null) {
            final SLinkedList b2 = list.b;
            list.b = b;
            list = (b = b2);
        }
        b.a(null);
    }
}
