// 
// Decompiled by Procyon v0.5.29
// 

public class EightQueens
{
    private int[] a;
    private boolean[] b;
    private int c;
    
    public static void main(final String[] array) {
        final EightQueens eightQueens;
        (eightQueens = new EightQueens(5)).a(0);
        System.out.println("There were " + eightQueens.c + " solutions.");
    }
    
    private EightQueens(int i) {
        this.a = new int[5];
        this.b = new boolean[5];
        this.c = 0;
        for (i = 0; i < 5; ++i) {
            this.a[i] = -1;
            this.b[i] = false;
        }
    }
    
    private void a(final int n) {
        if (n == this.a.length) {
            System.out.println("Here is a solution:\n");
            for (int i = 0; i < this.a.length; ++i) {
                for (int j = 0; j < this.a.length; ++j) {
                    if (this.a[j] == i) {
                        System.out.print("Q ");
                    }
                    else {
                        System.out.print("_ ");
                    }
                }
                System.out.println("\n");
            }
            ++this.c;
        }
        for (int k = 0; k < this.a.length; ++k) {
            if (!this.b[k]) {
                final int n2 = k;
                int n3 = 0;
                while (true) {
                    if (n3 < n) {
                        if (Math.abs(n - n3) == Math.abs(this.a[n3] - n2)) {
                            break;
                        }
                        ++n3;
                    }
                }
            }
        }
    }
}
