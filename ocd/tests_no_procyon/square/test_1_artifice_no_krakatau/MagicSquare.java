// 
// Decompiled by Procyon v0.5.29
// 

public class MagicSquare
{
    private int[][] f00;
    private boolean[] f10;
    private int f20;
    private int f30;
    private int f40;
    
    public static void main(final String[] array) {
        final MagicSquare magicSquare = new MagicSquare(3);
        magicSquare.m00(0, 0);
        System.out.println("There were " + magicSquare.m40() + " number of solutions.");
    }
    
    public MagicSquare(final int n) {
        this.f00 = new int[n][n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                this.f00[i][j] = 0;
            }
        }
        this.f20 = n * n;
        this.f10 = new boolean[this.f20];
        for (int k = 0; k < this.f20; ++k) {
            this.f10[k] = true;
        }
        this.f30 = n * (n * n + 1) / 2;
        this.f40 = 0;
    }
    
    public void m00(final int n, final int n2) {
        if (!this.m10() || !this.m20() || !this.m30()) {
            return;
        }
        if (n != this.f00.length) {
            for (int i = 0; i < this.f20; ++i) {
                if (this.f10[i]) {
                    this.f00[n][n2] = i + 1;
                    this.f10[i] = false;
                    final int n3 = n2 + 1;
                    int n4;
                    int n5;
                    if (n3 != this.f00.length) {
                        n4 = n3;
                        n5 = n;
                    }
                    else {
                        final int n6 = n + 1;
                        n4 = 0;
                        n5 = n6;
                    }
                    this.m00(n5, n4);
                    this.f00[n][n2] = 0;
                    this.f10[i] = true;
                }
            }
            return;
        }
        System.out.println(this);
        ++this.f40;
    }
    
    public boolean m10() {
        for (int i = 0; i < this.f00.length; ++i) {
            int n = 0;
            boolean b = false;
            int n3;
            for (int j = 0; j < this.f00[i].length; j = n3) {
                final int n2 = n + this.f00[i][j];
                final boolean b2 = (this.f00[i][j] != 0) ? b : true;
                n3 = j + 1;
                n = n2;
                b = b2;
            }
            if (!b && n != this.f30) {
                return false;
            }
        }
        return true;
    }
    
    public boolean m20() {
        for (int i = 0; i < this.f00[0].length; ++i) {
            int n = 0;
            boolean b = false;
            int n3;
            for (int j = 0; j < this.f00.length; j = n3) {
                final int n2 = n + this.f00[j][i];
                final boolean b2 = (this.f00[j][i] != 0) ? b : true;
                n3 = j + 1;
                n = n2;
                b = b2;
            }
            if (!b && n != this.f30) {
                return false;
            }
        }
        return true;
    }
    
    public boolean m30() {
        int n = 0;
        boolean b = false;
        int n3;
        for (int i = 0; i < this.f00.length; i = n3) {
            final int n2 = n + this.f00[i][i];
            final boolean b2 = (this.f00[i][i] != 0) ? b : true;
            n3 = i + 1;
            n = n2;
            b = b2;
        }
        if (!b && n != this.f30) {
            return false;
        }
        int n4 = 0;
        boolean b3 = false;
        int n6;
        for (int j = 0; j < this.f00.length; j = n6) {
            final int n5 = n4 + this.f00[j][this.f00.length - 1 - j];
            final boolean b4 = (this.f00[j][this.f00.length - 1 - j] != 0) ? b3 : true;
            n6 = j + 1;
            n4 = n5;
            b3 = b4;
        }
        return b3 || n4 == this.f30;
    }
    
    public int m40() {
        return this.f40;
    }
    
    @Override
    public String toString() {
        String s = "";
        int n2;
        for (int i = 0; i < this.f00.length; i = n2) {
            String s2 = s;
            int n;
            for (int j = 0; j < this.f00[i].length; j = n) {
                final String string = s2 + (0 + this.f00[i][j]) + "\t";
                n = j + 1;
                s2 = string;
            }
            final String string2 = s2 + "0\n";
            n2 = i + 1;
            s = string2;
        }
        return s;
    }
}
