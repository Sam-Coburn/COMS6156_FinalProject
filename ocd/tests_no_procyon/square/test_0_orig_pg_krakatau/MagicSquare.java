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
        final MagicSquare magicSquare = new MagicSquare(3);
        magicSquare.a(0, 0);
        System.out.println("There were " + magicSquare.e + " number of solutions.");
    }
    
    private MagicSquare(final int n) {
        this.a = new int[3][3];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                this.a[i][j] = 0;
            }
        }
        this.c = 9;
        this.b = new boolean[this.c];
        for (int k = 0; k < this.c; ++k) {
            this.b[k] = true;
        }
        this.d = 15;
        this.e = 0;
    }
    
    private void a(final int n, final int n2) {
        while (true) {
            for (int i = 0; i < this.a.length; ++i) {
                int n3 = 0;
                boolean b = false;
                int n5;
                for (int j = 0; j < this.a[i].length; j = n5) {
                    final int n4 = n3 + this.a[i][j];
                    final boolean b2 = (this.a[i][j] != 0) ? b : true;
                    n5 = j + 1;
                    n3 = n4;
                    b = b2;
                }
                if (!b) {
                    if (n3 != this.d) {
                        final boolean b3 = false;
                        if (b3) {
                            while (true) {
                                for (int k = 0; k < this.a[0].length; ++k) {
                                    int n6 = 0;
                                    boolean b4 = false;
                                    int n8;
                                    for (int l = 0; l < this.a.length; l = n8) {
                                        final int n7 = n6 + this.a[l][k];
                                        final boolean b5 = (this.a[l][k] != 0) ? b4 : true;
                                        n8 = l + 1;
                                        n6 = n7;
                                        b4 = b5;
                                    }
                                    if (!b4) {
                                        if (n6 != this.d) {
                                            final boolean b6 = false;
                                            if (!b6) {
                                                return;
                                            }
                                            int n9 = 0;
                                            boolean b7 = false;
                                            int n12;
                                            for (int n10 = 0; n10 < this.a.length; n10 = n12) {
                                                final int n11 = n9 + this.a[n10][n10];
                                                final boolean b8 = (this.a[n10][n10] != 0) ? b7 : true;
                                                n12 = n10 + 1;
                                                n9 = n11;
                                                b7 = b8;
                                            }
                                            boolean b9 = false;
                                            Label_0497: {
                                                if (!b7) {
                                                    if (n9 != this.d) {
                                                        b9 = false;
                                                        break Label_0497;
                                                    }
                                                }
                                                int n13 = 0;
                                                boolean b10 = false;
                                                int n16;
                                                for (int n14 = 0; n14 < this.a.length; n14 = n16) {
                                                    final int n15 = n13 + this.a[n14][this.a.length - 1 - n14];
                                                    final boolean b11 = (this.a[n14][this.a.length - 1 - n14] != 0) ? b10 : true;
                                                    n16 = n14 + 1;
                                                    n13 = n15;
                                                    b10 = b11;
                                                }
                                                if (!b10) {
                                                    if (n13 != this.d) {
                                                        b9 = false;
                                                        break Label_0497;
                                                    }
                                                }
                                                b9 = true;
                                            }
                                            if (!b9) {
                                                return;
                                            }
                                            if (n != this.a.length) {
                                                for (int n17 = 0; n17 < this.c; ++n17) {
                                                    if (this.b[n17]) {
                                                        this.a[n][n2] = n17 + 1;
                                                        this.b[n17] = false;
                                                        final int n18 = n2 + 1;
                                                        int n19;
                                                        int n20;
                                                        if (n18 != this.a.length) {
                                                            n19 = n18;
                                                            n20 = n;
                                                        }
                                                        else {
                                                            final int n21 = n + 1;
                                                            n19 = 0;
                                                            n20 = n21;
                                                        }
                                                        this.a(n20, n19);
                                                        this.a[n][n2] = 0;
                                                        this.b[n17] = true;
                                                    }
                                                }
                                                return;
                                            }
                                            System.out.println(this);
                                            ++this.e;
                                            return;
                                        }
                                    }
                                }
                                final boolean b6 = true;
                                continue;
                            }
                        }
                        return;
                    }
                }
            }
            final boolean b3 = true;
            continue;
        }
    }
    
    @Override
    public String toString() {
        String s = "";
        int n2;
        for (int i = 0; i < this.a.length; i = n2) {
            String s2 = s;
            int n;
            for (int j = 0; j < this.a[i].length; j = n) {
                final String string = s2 + this.a[i][j] + "\t";
                n = j + 1;
                s2 = string;
            }
            final String string2 = s2 + "\n";
            n2 = i + 1;
            s = string2;
        }
        return s;
    }
}
