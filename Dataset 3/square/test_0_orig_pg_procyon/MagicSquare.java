// 
// Decompiled by Procyon v0.5.29
// 

public class MagicSquare
{
    private int[][] a;
    private boolean[] b;
    private int c;
    private int d;
    private int e;
    
    public static void main(final String[] array) {
        final MagicSquare magicSquare;
        (magicSquare = new MagicSquare(3)).a(0, 0);
        System.out.println("There were " + magicSquare.e + " number of solutions.");
    }
    
    private MagicSquare(int i) {
        this.a = new int[3][3];
        int j;
        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                this.a[i][j] = 0;
            }
        }
        this.c = 9;
        this.b = new boolean[this.c];
        for (i = 0; i < this.c; ++i) {
            this.b[i] = true;
        }
        this.d = 15;
        this.e = 0;
    }
    
    private void a(final int n, final int n2) {
    Block_4:
        while (true) {
            for (int i = 0; i < this.a.length; ++i) {
                int n3 = 0;
                boolean b = false;
                for (int j = 0; j < this.a[i].length; ++j) {
                    n3 += this.a[i][j];
                    if (this.a[i][j] == 0) {
                        b = true;
                    }
                }
                if (!b && n3 != this.d) {
                    break Block_4;
                }
            }
        }
    }
    
    @Override
    public String toString() {
        String s = "";
        for (int i = 0; i < this.a.length; ++i) {
            for (int j = 0; j < this.a[i].length; ++j) {
                s = s + this.a[i][j] + "\t";
            }
            s += "\n";
        }
        return s;
    }
}
